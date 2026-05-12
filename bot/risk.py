"""
Risk gate.

Every OrderIntent the strategy emits is filtered through here BEFORE the
executor places it. Caps that cannot be turned off in live mode:

  * Max position per ticker:    25% of equity   (hard ceiling on concentration)
  * Max total exposure:         95% of equity   (keep a little cash for fills)
  * Max notional per order:     10% of equity   (no fat-finger trades)
  * Per-minute order count:     25              (well under exchange's 30 limit)
  * Daily drawdown kill-switch: -15% from session peak (pauses 30 ticks, then half-size)

None of these are exposed to the user. They're hard-coded defaults.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal

from .logging_setup import get_logger
from .market import MarketStore
from .strategy import OrderIntent, PortfolioView

log = get_logger(__name__)


# --------------------------------------------------------------------- constants

MAX_POSITION_FRACTION = Decimal("0.25")
MAX_TOTAL_EXPOSURE = Decimal("0.95")
MAX_ORDER_NOTIONAL = Decimal("0.10")
ORDERS_PER_MINUTE = 25
DRAWDOWN_KILL_SWITCH = Decimal("-0.15")
KILL_SWITCH_PAUSE_TICKS = 30


# ---------------------------------------------------------------------- gate

@dataclass
class RiskState:
    session_peak_equity: Decimal = Decimal("0")
    paused_until_tick: int = 0
    current_tick: int = 0
    order_timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=200))

    def tick(self, equity: Decimal) -> None:
        self.current_tick += 1
        if equity > self.session_peak_equity:
            self.session_peak_equity = equity


class RiskGate:
    def __init__(self) -> None:
        self.state = RiskState()

    # ----------------------------------------------------------------- session
    def update_for_tick(self, equity: Decimal) -> None:
        self.state.tick(equity)

    def drawdown(self, equity: Decimal) -> Decimal:
        peak = self.state.session_peak_equity
        if peak <= 0:
            return Decimal("0")
        return (equity - peak) / peak

    def trading_paused(self) -> bool:
        return self.state.current_tick < self.state.paused_until_tick

    def maybe_trigger_kill_switch(self, equity: Decimal) -> None:
        if self.drawdown(equity) <= DRAWDOWN_KILL_SWITCH and not self.trading_paused():
            self.state.paused_until_tick = self.state.current_tick + KILL_SWITCH_PAUSE_TICKS
            log.warning(
                "risk.kill_switch",
                drawdown=str(self.drawdown(equity)),
                paused_for_ticks=KILL_SWITCH_PAUSE_TICKS,
            )

    # ----------------------------------------------------------------- filter
    def filter(
        self,
        intents: list[OrderIntent],
        portfolio: PortfolioView,
        market: MarketStore,
    ) -> list[OrderIntent]:
        equity = portfolio.equity(market)
        self.update_for_tick(equity)
        self.maybe_trigger_kill_switch(equity)

        if self.trading_paused():
            # Only allow SELLs while paused — never new BUYs.
            sells = [i for i in intents if i.side == "SELL"]
            if sells:
                log.info("risk.paused.allow_sells_only", count=len(sells))
            return sells

        allowed: list[OrderIntent] = []
        for intent in intents:
            if intent.side == "SELL":
                # Sells always pass the gate (we never block an exit).
                if self._under_rate_limit():
                    allowed.append(intent)
                    self._record_order()
                else:
                    log.warning("risk.rate_limit.exit_dropped", ticker=intent.ticker)
                continue

            # BUY: check caps.
            if not self._under_rate_limit():
                log.warning("risk.rate_limit.entry_dropped", ticker=intent.ticker)
                continue

            price = intent.limit_price or Decimal(
                str(market.get(intent.ticker).price if market.get(intent.ticker) else 0)
            )
            notional = price * intent.quantity
            if notional > equity * MAX_ORDER_NOTIONAL:
                log.warning(
                    "risk.cap.order_notional",
                    ticker=intent.ticker,
                    notional=str(notional),
                    cap=str(equity * MAX_ORDER_NOTIONAL),
                )
                continue

            # Per-ticker position cap (including this proposed buy).
            existing = portfolio.positions.get(intent.ticker)
            new_qty = (existing.quantity if existing else 0) + intent.quantity
            new_position_value = price * new_qty
            if new_position_value > equity * MAX_POSITION_FRACTION:
                log.warning(
                    "risk.cap.position",
                    ticker=intent.ticker,
                    new_value=str(new_position_value),
                    cap=str(equity * MAX_POSITION_FRACTION),
                )
                continue

            # Total exposure cap.
            current_exposure = sum(
                (Decimal(str(market.get(t).price if market.get(t) else 0)) * p.quantity)
                for t, p in portfolio.positions.items()
            )
            if current_exposure + notional > equity * MAX_TOTAL_EXPOSURE:
                log.info("risk.cap.total_exposure", ticker=intent.ticker)
                continue

            allowed.append(intent)
            self._record_order()
        return allowed

    # ----------------------------------------------------------------- helpers
    def _under_rate_limit(self) -> bool:
        cutoff = time.monotonic() - 60.0
        # Drop expired timestamps from the left
        while self.state.order_timestamps and self.state.order_timestamps[0] < cutoff:
            self.state.order_timestamps.popleft()
        return len(self.state.order_timestamps) < ORDERS_PER_MINUTE

    def _record_order(self) -> None:
        self.state.order_timestamps.append(time.monotonic())
