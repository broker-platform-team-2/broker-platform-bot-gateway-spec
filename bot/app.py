"""
The orchestrator — v2.

Pipeline:
    PRICE_UPDATE      → market store → (every 12 msgs) strategy.decide()
                        → risk.filter() → executor.submit_all()
    ORDER_UPDATE      → executor.handle_order_update() + reconcile portfolio
    MARKET_EVENT      → events.react() → risk.filter() → executor.submit_all()
                        + schedule auto-exit when duration_ticks elapses
    ORDER_BOOK_UPDATE → orderbook cache (used by strategy for spread/imbalance)
    every 30 s        → reconcile portfolio + cash from /portfolio + /accounts/me

v2 additions:
  - orderbook passed to strategy.decide() for spread filter and imbalance signal
  - entry_atr threaded from OrderIntent → Position so exit logic has ATR context
  - event auto-expiry: positions opened on event are auto-sold after duration_ticks
  - current_tick passed to events.react() for debounce
  - peak_prices survive reconcile cycle (trailing stop levels persist)
"""
from __future__ import annotations

import asyncio
import signal
import time
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


# Run the decision pipeline at most once per N PRICE_UPDATE messages.
# With 8 tickers at 1 tick/sec the WS fires 8 msgs/sec; every 12 msgs ≈ 1.5 s.
DECISION_INTERVAL_PRICE_UPDATES = 12


class Bot:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.http      = HttpClient(settings)
        self.market    = MarketStore()
        self.orderbook = OrderBook()
        self.ws        = WsClient(settings, get_token=lambda: self.http.token)
        self.strategy  = Strategy()
        self.executor  = Executor(self.http)
        self.events    = EventReactor(self.strategy)

        # Per-subscriber state — keyed by userId
        self._subscriber_portfolios: dict[int, PortfolioView] = {}
        self._subscriber_risk_gates: dict[int, RiskGate]      = {}

        # Tracks sell quantities mid-flight so reconcile doesn't restore them.
        self._pending_sells: dict[int, dict[str, int]] = {}

        # Authoritative peak prices — survive the 30-s reconcile cycle.
        self._peak_prices: dict[int, dict[str, Decimal]] = {}

        # ATR values recorded at entry — used by _reconcile_subscriber to stamp
        # entry_atr on Position objects so exit logic has the right stop level.
        self._entry_atrs: dict[int, dict[str, Decimal]] = {}

        # Event-driven position expiry.
        # Keyed by (user_id, ticker) → monotonic expiry time.
        # When time.monotonic() exceeds the value we emit an auto-sell.
        # duration_ticks from the exchange ≈ real seconds (tick-rate-ms = 1000).
        self._event_expires: dict[int, dict[str, float]] = {}

        self._price_update_counter = 0
        self._decision_lock        = asyncio.Lock()
        self._reconcile_task: asyncio.Task[None] | None = None

    # --------------------------------------------------------------- bootstrap

    async def bootstrap(self) -> None:
        await self.http.authenticate()
        await self._snapshot_market()
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
        """Pull cash + positions from the platform and rebuild PortfolioView."""
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

        existing        = self._subscriber_portfolios.get(user_id)
        user_peaks      = self._peak_prices.get(user_id, {})
        user_entry_atrs = self._entry_atrs.get(user_id, {})

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
            peak      = user_peaks.get(ticker) or (existing_pos.peak_price if existing_pos else avg)
            # Restore entry_atr from our internal store so exit logic keeps
            # the ATR that was recorded at the moment of entry, not recalculated.
            entry_atr = user_entry_atrs.get(ticker, Decimal("0"))

            new_positions[ticker] = Position(
                ticker=ticker, quantity=qty, avg_cost=avg,
                peak_price=peak, entry_atr=entry_atr,
            )

        # Subtract pending sells — don't let reconcile undo an in-flight sell.
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
                        entry_atr=pos.entry_atr,
                    )
            else:
                pending_sells.pop(ticker, None)

        # Drop stale peak / atr entries for positions closed on the server.
        open_tickers = set(new_positions)
        for store in (self._peak_prices.get(user_id, {}),
                      self._entry_atrs.get(user_id, {})):
            for stale in [t for t in list(store) if t not in open_tickers]:
                store.pop(stale, None)

        portfolio = PortfolioView(cash=cash, positions=new_positions)
        self._subscriber_portfolios[user_id] = portfolio
        log.info(
            "reconcile.ok", user_id=user_id, cash=str(cash),
            positions=len(new_positions), equity=str(portfolio.equity(self.market)),
        )

    async def _reconcile_loop(self) -> None:
        while True:
            await asyncio.sleep(30.0)
            for sub in await self.http.get_active_subscribers():
                await self._reconcile_subscriber(sub["userId"])

    # ----------------------------------------------------------------- handlers

    def _register_handlers(self) -> None:
        self.ws.on("PRICE_UPDATE",      self._on_price_update)
        self.ws.on("ORDER_UPDATE",      self._on_order_update)
        self.ws.on("MARKET_EVENT",      self._on_market_event)
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

            # Inject auto-sell intents for event-driven positions that have expired
            expiry_intents = self._collect_event_expiry_intents(user_id, portfolio)

            # Strategy decision — now includes orderbook for spread/imbalance
            intents = self.strategy.decide(self.market, portfolio, orderbook=self.orderbook)
            intents = expiry_intents + intents

            # Persist peak prices updated by _exit_decision so they survive
            # the next reconcile cycle.
            peaks = self._peak_prices.setdefault(user_id, {})
            for ticker, pos in portfolio.positions.items():
                peaks[ticker] = pos.peak_price

            if not intents:
                continue

            risk    = self._subscriber_risk_gates.setdefault(user_id, RiskGate())
            allowed = risk.filter(intents, portfolio, self.market)
            if not allowed:
                continue

            log.info(
                "decisions", user_id=user_id,
                proposed=len(intents), allowed=len(allowed),
                actions=[{
                    "side": i.side, "ticker": i.ticker,
                    "qty": i.quantity, "reason": i.reason,
                } for i in allowed],
            )
            await self.executor.submit_all(allowed, on_behalf_of=user_id)
            self._apply_optimistic_update(user_id, allowed)

    async def _on_order_update(self, payload: dict[str, Any]) -> None:
        self.executor.handle_order_update(payload)
        status = payload.get("status")
        if status in ("FILLED", "PARTIALLY_FILLED", "CANCELLED", "REJECTED", "EXPIRED"):
            for sub in await self.http.get_active_subscribers():
                await self._reconcile_subscriber(sub["userId"])

    async def _on_market_event(self, payload: dict[str, Any]) -> None:
        # Refresh scores so EventReactor picks tickers on current data.
        self.strategy._score_all(self.market, orderbook=self.orderbook)  # noqa: SLF001

        subscribers = await self.http.get_active_subscribers()
        if not subscribers:
            return

        event_type = payload.get("event_type") or payload.get("eventType")
        log.info(
            "market_event",
            event_type=event_type,
            headline=payload.get("headline"),
            scope=payload.get("scope"),
            target=payload.get("target"),
            magnitude=payload.get("magnitude"),
            duration_ticks=payload.get("duration_ticks"),
        )

        for sub in subscribers:
            user_id   = sub["userId"]
            portfolio = self._subscriber_portfolios.get(
                user_id, PortfolioView(cash=Decimal("0"))
            )

            # Pass current_tick for debounce; returns (intents, duration_ticks)
            intents, duration = self.events.react(
                payload, self.market, portfolio,
                current_tick=self._price_update_counter,
            )
            if not intents:
                continue

            risk    = self._subscriber_risk_gates.setdefault(user_id, RiskGate())
            allowed = risk.filter(intents, portfolio, self.market)
            if not allowed:
                continue

            log.info(
                "event.actions", user_id=user_id, event_type=event_type,
                proposed=len(intents), allowed=len(allowed),
            )
            await self.executor.submit_all(allowed, on_behalf_of=user_id)
            self._apply_optimistic_update(user_id, allowed)

            # Schedule auto-exit for event-driven BUY positions.
            # duration is in simulation ticks (1 tick ≈ 1 real second).
            if duration > 0:
                expire_at = time.monotonic() + float(duration)
                user_expires = self._event_expires.setdefault(user_id, {})
                for intent in allowed:
                    if intent.side == "BUY":
                        user_expires[intent.ticker] = expire_at
                        log.info(
                            "event.expiry_scheduled",
                            ticker=intent.ticker,
                            expire_in_s=duration,
                        )

    async def _on_order_book_update(self, payload: dict[str, Any]) -> None:
        self.orderbook.apply_update(payload)
        log.debug("orderbook.updated", ticker=payload.get("ticker") or payload.get("symbol"))

    # --------------------------------------------------------- event expiry

    def _collect_event_expiry_intents(
        self,
        user_id: int,
        portfolio: PortfolioView,
    ) -> list:
        """Return SELL intents for event-driven positions whose duration has elapsed."""
        from .strategy import OrderIntent  # local to avoid circular at module level

        user_expires = self._event_expires.get(user_id)
        if not user_expires:
            return []

        now     = time.monotonic()
        expired = [t for t, exp in user_expires.items() if now >= exp]
        intents = []
        for ticker in expired:
            del user_expires[ticker]
            pos = portfolio.positions.get(ticker)
            if pos and pos.quantity > 0:
                log.info("event.expiry_sell", ticker=ticker, user_id=user_id)
                intents.append(OrderIntent(
                    side="SELL", ticker=ticker, quantity=pos.quantity,
                    order_type="MARKET", reason="event_expired",
                ))
        return intents

    # ------------------------------------------------- optimistic update

    def _apply_optimistic_update(self, user_id: int, intents: list) -> None:
        portfolio = self._subscriber_portfolios.get(user_id)
        if portfolio is None:
            return

        new_positions = dict(portfolio.positions)
        pending    = self._pending_sells.setdefault(user_id, {})
        peaks      = self._peak_prices.setdefault(user_id, {})
        entry_atrs = self._entry_atrs.setdefault(user_id, {})

        for intent in intents:
            if intent.side == "SELL" and intent.ticker in new_positions:
                pos     = new_positions[intent.ticker]
                new_qty = pos.quantity - intent.quantity
                if new_qty <= 0:
                    del new_positions[intent.ticker]
                    peaks.pop(intent.ticker, None)
                    entry_atrs.pop(intent.ticker, None)
                else:
                    new_positions[intent.ticker] = Position(
                        ticker=pos.ticker, quantity=new_qty,
                        avg_cost=pos.avg_cost, peak_price=pos.peak_price,
                        entry_atr=pos.entry_atr,
                    )
                pending[intent.ticker] = pending.get(intent.ticker, 0) + intent.quantity

            elif intent.side == "BUY":
                state = self.market.get(intent.ticker)
                price = (
                    Decimal(str(state.price)) if state
                    else (intent.limit_price or Decimal("0"))
                )
                if intent.entry_atr > 0:
                    entry_atrs[intent.ticker] = intent.entry_atr

                if intent.ticker in new_positions:
                    pos       = new_positions[intent.ticker]
                    total_qty = pos.quantity + intent.quantity
                    new_avg   = (pos.avg_cost * pos.quantity + price * intent.quantity) / total_qty
                    new_positions[intent.ticker] = Position(
                        ticker=pos.ticker, quantity=total_qty,
                        avg_cost=new_avg, peak_price=max(pos.peak_price, price),
                        entry_atr=pos.entry_atr or intent.entry_atr,
                    )
                else:
                    new_positions[intent.ticker] = Position(
                        ticker=intent.ticker, quantity=intent.quantity,
                        avg_cost=price, peak_price=price,
                        entry_atr=intent.entry_atr,
                    )
                peaks[intent.ticker] = max(peaks.get(intent.ticker, price), price)

        self._subscriber_portfolios[user_id] = PortfolioView(
            cash=portfolio.cash, positions=new_positions,
        )

    # ---------------------------------------------------------------- run loop

    async def run(self) -> None:
        configure_logging(self.settings.log_level)
        log.info("bot.start", gateway=self.settings.gateway_http_url,
                 email=self.settings.bot_email)
        await self.bootstrap()

        if recording_enabled():
            recorder = open_run_file()
            self.ws.set_recorder(recorder)
            log.info("replay.recording", path=str(recorder.path))
        else:
            recorder = None

        await self.ws.start()
        self._reconcile_task = asyncio.create_task(
            self._reconcile_loop(), name="reconcile"
        )

        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except NotImplementedError:
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
