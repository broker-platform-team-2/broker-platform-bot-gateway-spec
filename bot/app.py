"""
The orchestrator.

Day 2 wires the full pipeline:

    PRICE_UPDATE  ->  strategy.decide()  ->  risk.filter()  ->  executor.submit_all()
    ORDER_UPDATE  ->  executor.handle_order_update()  +  refresh portfolio
    every 30 s    ->  reconcile portfolio + cash from /portfolio + /accounts/me

Day 3 will hook events.MARKET_EVENT and pyramiding in here too.

Track B additions:
  - ORDER_BOOK_UPDATE is now forwarded to an OrderBook cache; strategy uses
    real best-ask prices for LIMIT entries when book data is available.
  - When REPLAY_RECORD=1, a RunRecorder is attached to WsClient so every
    incoming message is written to runs/<timestamp>.jsonl for later replay.
"""
from __future__ import annotations

import asyncio
import signal
from decimal import Decimal
from typing import Any

from .config import Settings
from .events import EventReactor
from .executor import Executor
from .http_client import HttpClient
from .logging_setup import configure as configure_logging, get_logger
from .market import MarketStore
from .orderbook import OrderBook
from .replay import open_run_file, recording_enabled
from .risk import RiskGate
from .strategy import PortfolioView, Position, Strategy
from .ws_client import WsClient

log = get_logger(__name__)


# Run the decision pipeline at most once per N PRICE_UPDATE messages so we
# don't spam the strategy with every single ticker change (the WS sends one
# message per ticker per tick).
DECISION_INTERVAL_PRICE_UPDATES = 12


class Bot:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.http = HttpClient(settings)
        self.market = MarketStore()
        self.orderbook = OrderBook()
        self.ws = WsClient(settings, get_token=lambda: self.http.token)
        self.strategy = Strategy()
        self.executor = Executor(self.http)
        self.events = EventReactor(self.strategy)

        # Per-subscriber state — keyed by userId
        self._subscriber_portfolios: dict[int, PortfolioView] = {}
        self._subscriber_risk_gates: dict[int, RiskGate] = {}
        # Tracks quantities being sold that have not yet settled on the server.
        # Reconcile subtracts these so it never restores a position mid-flight.
        self._pending_sells: dict[int, dict[str, int]] = {}

        self._price_update_counter = 0
        self._decision_lock = asyncio.Lock()
        self._reconcile_task: asyncio.Task[None] | None = None

    # --------------------------------------------------------------- bootstrap
    async def bootstrap(self) -> None:
        await self.http.authenticate()
        await self._snapshot_market()
        # Prime per-subscriber portfolios
        for sub in await self.http.get_active_subscribers():
            await self._reconcile_subscriber(sub["userId"])
        self._register_handlers()

    async def _snapshot_market(self) -> None:
        try:
            stocks = await self.http.get_market_snapshot()
            self.market.seed_from_snapshot(stocks)
            log.info("market.snapshot", tickers=len(self.market))
        except Exception as exc:  # noqa: BLE001
            log.warning("market.snapshot.failed", error=str(exc))

    async def _reconcile_subscriber(self, user_id: int) -> None:
        """Pull cash + positions for one subscriber from the platform."""
        try:
            accounts = await self.http.get_accounts(on_behalf_of=user_id)
            holdings = await self.http.get_portfolio(on_behalf_of=user_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("reconcile.subscriber.failed", user_id=user_id, error=str(exc))
            return

        target_currency = self.settings.seed_currency.upper()
        cash = Decimal("0")
        for acc in accounts:
            if (acc.get("currency") or "").upper() == target_currency:
                try:
                    cash = Decimal(str(acc.get("balance") or acc.get("available") or "0"))
                except Exception:  # noqa: BLE001
                    cash = Decimal("0")
                break

        existing = self._subscriber_portfolios.get(user_id)
        new_positions: dict[str, Position] = {}
        for h in holdings:
            ticker = h.get("instrumentId") or h.get("ticker")
            if not ticker:
                continue
            try:
                qty = int(h.get("amount") or h.get("quantity") or 0)
                avg = Decimal(str(h.get("averageCost") or h.get("average_cost") or "0"))
            except Exception:  # noqa: BLE001
                continue
            if qty <= 0:
                continue
            existing_pos = existing.positions.get(ticker) if existing else None
            peak = existing_pos.peak_price if existing_pos else avg
            new_positions[ticker] = Position(ticker=ticker, quantity=qty, avg_cost=avg, peak_price=peak)

        # Subtract any pending sells so a mid-flight order doesn't restore
        # a position that we already placed a sell for but hasn't settled yet.
        pending_sells = self._pending_sells.get(user_id, {})
        for ticker, pending_qty in list(pending_sells.items()):
            if ticker in new_positions:
                adjusted = new_positions[ticker].quantity - pending_qty
                if adjusted <= 0:
                    del new_positions[ticker]
                else:
                    pos = new_positions[ticker]
                    new_positions[ticker] = Position(
                        ticker=pos.ticker, quantity=adjusted,
                        avg_cost=pos.avg_cost, peak_price=pos.peak_price,
                    )
            else:
                # Server confirms position is gone — order settled, clear pending entry
                pending_sells.pop(ticker, None)

        portfolio = PortfolioView(cash=cash, positions=new_positions)
        self._subscriber_portfolios[user_id] = portfolio
        log.info("reconcile.ok", user_id=user_id, cash=str(cash),
                 positions=len(new_positions), equity=str(portfolio.equity(self.market)))

    async def _reconcile_loop(self) -> None:
        while True:
            await asyncio.sleep(30.0)
            for sub in await self.http.get_active_subscribers():
                await self._reconcile_subscriber(sub["userId"])

    # ----------------------------------------------------------------- handlers
    def _register_handlers(self) -> None:
        self.ws.on("PRICE_UPDATE", self._on_price_update)
        self.ws.on("ORDER_UPDATE", self._on_order_update)
        self.ws.on("MARKET_EVENT", self._on_market_event)
        self.ws.on("ORDER_BOOK_UPDATE", self._on_order_book_update)

    async def _on_price_update(self, payload: dict[str, Any]) -> None:
        self.market.apply_price_update(payload)
        self._price_update_counter += 1
        if self._price_update_counter % DECISION_INTERVAL_PRICE_UPDATES != 0:
            return
        if self._decision_lock.locked():
            return
        async with self._decision_lock:
            await self._run_decision_cycle()

    async def _run_decision_cycle(self) -> None:
        subscribers = await self.http.get_active_subscribers()
        if not subscribers:
            log.info("bot.paused", reason="no active subscribers")
            return

        for sub in subscribers:
            user_id = sub["userId"]
            if user_id not in self._subscriber_portfolios:
                await self._reconcile_subscriber(user_id)
            portfolio = self._subscriber_portfolios.get(user_id)
            if portfolio is None:
                continue

            intents = self.strategy.decide(self.market, portfolio)
            if not intents:
                continue

            risk = self._subscriber_risk_gates.setdefault(user_id, RiskGate())
            allowed = risk.filter(intents, portfolio, self.market)
            if not allowed:
                continue

            log.info("decisions", user_id=user_id, proposed=len(intents), allowed=len(allowed),
                     actions=[{"side": i.side, "ticker": i.ticker,
                                "qty": i.quantity, "reason": i.reason} for i in allowed])
            await self.executor.submit_all(allowed, on_behalf_of=user_id)
            self._apply_optimistic_update(user_id, allowed)

    async def _on_order_update(self, payload: dict[str, Any]) -> None:
        self.executor.handle_order_update(payload)
        status = payload.get("status")
        if status in ("FILLED", "PARTIALLY_FILLED", "CANCELLED", "REJECTED", "EXPIRED"):
            for sub in await self.http.get_active_subscribers():
                await self._reconcile_subscriber(sub["userId"])

    async def _on_market_event(self, payload: dict[str, Any]) -> None:
        subscribers = await self.http.get_active_subscribers()
        if not subscribers:
            return

        log.info("market_event",
                 event_type=payload.get("event_type") or payload.get("eventType"),
                 headline=payload.get("headline"),
                 scope=payload.get("scope"),
                 target=payload.get("target"))

        self.strategy._score_all(self.market)  # noqa: SLF001

        for sub in subscribers:
            user_id = sub["userId"]
            portfolio = self._subscriber_portfolios.get(user_id, PortfolioView(cash=Decimal("0")))
            intents = self.events.react(payload, self.market, portfolio)
            if not intents:
                continue
            risk = self._subscriber_risk_gates.setdefault(user_id, RiskGate())
            allowed = risk.filter(intents, portfolio, self.market)
            if not allowed:
                continue
            log.info("event.actions", user_id=user_id,
                     event_type=payload.get("event_type") or payload.get("eventType"),
                     proposed=len(intents), allowed=len(allowed))
            await self.executor.submit_all(allowed, on_behalf_of=user_id)
            self._apply_optimistic_update(user_id, allowed)

    async def _on_order_book_update(self, payload: dict[str, Any]) -> None:
        self.orderbook.apply_update(payload)
        log.debug("orderbook.updated", ticker=payload.get("ticker") or payload.get("symbol"))

    def _apply_optimistic_update(self, user_id: int, intents: list) -> None:
        """Immediately reflect placed orders in local portfolio state.

        Also records sells in _pending_sells so that reconcile cannot restore
        a position while its sell order is still settling on the exchange.
        """
        portfolio = self._subscriber_portfolios.get(user_id)
        if portfolio is None:
            return
        new_positions = dict(portfolio.positions)
        pending = self._pending_sells.setdefault(user_id, {})
        for intent in intents:
            if intent.side == "SELL" and intent.ticker in new_positions:
                pos = new_positions[intent.ticker]
                new_qty = pos.quantity - intent.quantity
                if new_qty <= 0:
                    del new_positions[intent.ticker]
                else:
                    new_positions[intent.ticker] = Position(
                        ticker=pos.ticker, quantity=new_qty,
                        avg_cost=pos.avg_cost, peak_price=pos.peak_price,
                    )
                # Record pending so reconcile doesn't undo this
                pending[intent.ticker] = pending.get(intent.ticker, 0) + intent.quantity
        self._subscriber_portfolios[user_id] = PortfolioView(
            cash=portfolio.cash, positions=new_positions,
        )

    # ---------------------------------------------------------------- run loop
    async def run(self) -> None:
        configure_logging(self.settings.log_level)
        log.info(
            "bot.start",
            gateway=self.settings.gateway_http_url,
            email=self.settings.bot_email,
        )
        await self.bootstrap()

        if recording_enabled():
            recorder = open_run_file()
            self.ws.set_recorder(recorder)
            log.info("replay.recording", path=str(recorder.path))
        else:
            recorder = None

        await self.ws.start()
        self._reconcile_task = asyncio.create_task(self._reconcile_loop(), name="reconcile")

        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except NotImplementedError:
                # Windows asyncio doesn't support add_signal_handler - KeyboardInterrupt
                # bubbles up naturally there.
                pass

        try:
            await stop_event.wait()
        except KeyboardInterrupt:
            pass
        finally:
            log.info("bot.stopping")
            if self._reconcile_task:
                self._reconcile_task.cancel()
                try:
                    await self._reconcile_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            await self.ws.stop()
            await self.http.aclose()
            if recorder is not None:
                recorder.close()
                log.info("replay.recorded", path=str(recorder.path))
            log.info("bot.stopped")


async def main() -> None:
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    bot = Bot(settings)
    try:
        await bot.run()
    except KeyboardInterrupt:
        pass


def replay_main(jsonl_path: str) -> None:
    """Entry point for `python -m bot replay <file>`."""
    from .replay import replay
    replay(jsonl_path)
