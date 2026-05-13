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
        self.risk = RiskGate()
        self.executor = Executor(self.http)
        self.events = EventReactor(self.strategy)

        self.portfolio = PortfolioView(cash=Decimal("0"))
        self._price_update_counter = 0
        self._decision_lock = asyncio.Lock()
        self._reconcile_task: asyncio.Task[None] | None = None

    # --------------------------------------------------------------- bootstrap
    async def bootstrap(self) -> None:
        await self.http.authenticate()
        await self._ensure_funded()
        await self._reconcile_portfolio()
        await self._snapshot_market()
        self._register_handlers()

    async def _ensure_funded(self) -> None:
        accounts = await self.http.get_accounts()
        target_currency = self.settings.seed_currency.upper()
        balance = Decimal("0")
        for acc in accounts:
            ccy = (acc.get("currency") or "").upper()
            if ccy == target_currency:
                bal_raw = acc.get("balance") or acc.get("available") or "0"
                try:
                    balance = Decimal(str(bal_raw))
                except Exception:  # noqa: BLE001
                    balance = Decimal("0")
                break

        if balance < self.settings.seed_deposit:
            top_up = self.settings.seed_deposit - balance
            log.info(
                "funds.topping_up",
                currency=target_currency,
                current=str(balance),
                deposit=str(top_up),
            )
            await self.http.deposit(target_currency, top_up)
        else:
            log.info("funds.ok", currency=target_currency, balance=str(balance))

    async def _snapshot_market(self) -> None:
        try:
            stocks = await self.http.get_market_snapshot()
            self.market.seed_from_snapshot(stocks)
            log.info("market.snapshot", tickers=len(self.market))
        except Exception as exc:  # noqa: BLE001
            log.warning("market.snapshot.failed", error=str(exc))

    async def _reconcile_portfolio(self) -> None:
        """Pull cash + positions from the platform - source of truth."""
        try:
            accounts = await self.http.get_accounts()
            holdings = await self.http.get_portfolio()
        except Exception as exc:  # noqa: BLE001
            log.warning("reconcile.failed", error=str(exc))
            return

        # Cash
        target_currency = self.settings.seed_currency.upper()
        cash = Decimal("0")
        for acc in accounts:
            if (acc.get("currency") or "").upper() == target_currency:
                try:
                    cash = Decimal(str(acc.get("balance") or acc.get("available") or "0"))
                except Exception:  # noqa: BLE001
                    cash = Decimal("0")
                break

        # Positions
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
            existing_peak = self.portfolio.positions.get(ticker)
            peak = existing_peak.peak_price if existing_peak else avg
            new_positions[ticker] = Position(
                ticker=ticker, quantity=qty, avg_cost=avg, peak_price=peak
            )

        self.portfolio = PortfolioView(cash=cash, positions=new_positions)
        log.info(
            "reconcile.ok",
            cash=str(cash),
            positions=len(new_positions),
            equity=str(self.portfolio.equity(self.market)),
        )

    async def _reconcile_loop(self) -> None:
        while True:
            await asyncio.sleep(30.0)
            await self._reconcile_portfolio()

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
        # Don't let two ticks of decisions run concurrently.
        if self._decision_lock.locked():
            return
        async with self._decision_lock:
            await self._run_decision_cycle()

    async def _run_decision_cycle(self) -> None:
        intents = self.strategy.decide(self.market, self.portfolio, self.orderbook)
        if not intents:
            return
        allowed = self.risk.filter(intents, self.portfolio, self.market)
        if not allowed:
            return
        log.info(
            "decisions",
            proposed=len(intents),
            allowed=len(allowed),
            actions=[
                {"side": i.side, "ticker": i.ticker, "qty": i.quantity, "reason": i.reason}
                for i in allowed
            ],
        )
        await self.executor.submit_all(allowed)

    async def _on_order_update(self, payload: dict[str, Any]) -> None:
        self.executor.handle_order_update(payload)
        status = payload.get("status")
        if status in ("FILLED", "PARTIALLY_FILLED", "CANCELLED", "REJECTED", "EXPIRED"):
            # Fast-path reconcile so the next decision cycle sees the new state.
            await self._reconcile_portfolio()

    async def _on_market_event(self, payload: dict[str, Any]) -> None:
        # Track A: event reactor runs immediately — speed matters more than
        # confirmation here. Intents still flow through the risk gate.
        log.info(
            "market_event",
            event_type=payload.get("event_type") or payload.get("eventType"),
            headline=payload.get("headline"),
            scope=payload.get("scope"),
            target=payload.get("target"),
        )
        # Refresh the strategy's score cache so the reactor's "top-N" ranks
        # are based on the latest data. This is cheap (~ms).
        self.strategy._score_all(self.market)  # noqa: SLF001 — same-package
        intents = self.events.react(payload, self.market, self.portfolio)
        if not intents:
            return
        allowed = self.risk.filter(intents, self.portfolio, self.market)
        if not allowed:
            return
        log.info(
            "event.actions",
            event_type=payload.get("event_type") or payload.get("eventType"),
            proposed=len(intents),
            allowed=len(allowed),
            actions=[
                {"side": i.side, "ticker": i.ticker, "qty": i.quantity, "reason": i.reason}
                for i in allowed
            ],
        )
        await self.executor.submit_all(allowed)

    async def _on_order_book_update(self, payload: dict[str, Any]) -> None:
        self.orderbook.apply_update(payload)
        log.debug("orderbook.updated", ticker=payload.get("ticker") or payload.get("symbol"))

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
