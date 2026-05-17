"""
Risk gate — v2.

Every OrderIntent the strategy emits is filtered through here BEFORE the
executor places it. Hard caps (never configurable):

  * Max position per ticker:    25% of equity
  * Max total exposure:         95% of equity
  * Max notional per order:     10% of equity
  * Per-sector exposure:        35% of equity
  * Max quantity per order:     281             (applies to both BUY and SELL)
  * Per-minute order count:     25              (exchange limit is 30)
  * Drawdown-proportional size: 4-band scaling
      0% to -5%    -> 1.0x  (full size)
      -5% to -10%  -> 0.65x (reduced)
      -10% to -15% -> 0.35x (survival mode)
      below -15%   -> 0.10x (recovery mode — tiny buys to dig out)
                             + 30-tick trading PAUSE on first breach

Kill-switch design (v2 fix):
  The old implementation returned 0x below -15%, which caused the bot to
  freeze permanently once all positions were closed: no buys possible,
  nothing to sell -> completely idle.

  The fix: after the 30-tick pause the bot resumes at 0.10x (recovery mode)
  so it can make small entries and work its way back.  The pause still
  provides the "stop and assess" moment; 0.10x means any single buy
  deploys at most 1% of equity (10% fraction x 0.10 multiplier), limiting
  further damage while allowing recovery.
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

MAX_POSITION_FRACTION  = Decimal("0.25")
MAX_TOTAL_EXPOSURE     = Decimal("0.95")
MAX_ORDER_NOTIONAL     = Decimal("0.10")
MAX_SECTOR_EXPOSURE    = Decimal("0.35")   # per-sector cap
ORDERS_PER_MINUTE      = 25
MAX_ORDER_QUANTITY     = 281

# Drawdown bands — thresholds and corresponding BUY size multipliers
DRAWDOWN_KILL_SWITCH   = Decimal("-0.15")  # below this -> pause then 0.10x recovery
DRAWDOWN_BAND_MEDIUM   = Decimal("-0.10")  # -10% to -15% -> 0.35x
DRAWDOWN_BAND_LIGHT    = Decimal("-0.05")  # -5%  to -10% -> 0.65x
SIZE_MULT_FULL         = Decimal("1.00")
SIZE_MULT_LIGHT        = Decimal("0.65")
SIZE_MULT_MEDIUM       = Decimal("0.35")
SIZE_MULT_RECOVERY     = Decimal("0.10")   # post-kill-switch recovery mode

# After kill-switch fires, trading is completely paused for this many ticks.
# Once the pause expires the bot resumes at SIZE_MULT_RECOVERY (not 0x).
KILL_SWITCH_PAUSE_TICKS = 30


# ---------------------------------------------------------------------- state

@dataclass
class RiskState:
    session_peak_equity: Decimal = Decimal("0")
    paused_until_tick: int = 0
    current_tick: int = 0
    order_timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    kill_switch_active: bool = False  # True while drawdown is below the kill-switch threshold

    def tick(self, equity: Decimal) -> None:
        self.current_tick += 1
        if equity > self.session_peak_equity:
            self.session_peak_equity = equity


# ---------------------------------------------------------------------- gate

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

    def _size_multiplier(self, equity: Decimal) -> Decimal:
        """Return the BUY quantity multiplier based on current drawdown band.

        Below DRAWDOWN_KILL_SWITCH the bot enters recovery mode (0.10x) rather
        than a hard 0x block.  The initial pause (KILL_SWITCH_PAUSE_TICKS) is
        enforced separately by trading_paused() / maybe_trigger_kill_switch().
        After the pause expires the bot can make tiny recovery entries so it
        isn't permanently frozen when flat and deep in drawdown.
        """
        dd = self.drawdown(equity)
        if dd <= DRAWDOWN_KILL_SWITCH:
            return SIZE_MULT_RECOVERY  # 0.10x — tiny recovery buys only
        if dd <= DRAWDOWN_BAND_MEDIUM:
            return SIZE_MULT_MEDIUM    # -10% to -15% -> 0.35x
        if dd <= DRAWDOWN_BAND_LIGHT:
            return SIZE_MULT_LIGHT     # -5%  to -10% -> 0.65x
        return SIZE_MULT_FULL          # 0%   to -5%  -> 1.00x

    def maybe_trigger_kill_switch(self, equity: Decimal) -> None:
        dd = self.drawdown(equity)
        if dd > DRAWDOWN_KILL_SWITCH:
            # Equity has recovered above the threshold — allow the kill switch
            # to fire again if we breach it in the future.
            self.state.kill_switch_active = False
            return
        if not self.state.kill_switch_active:
            # First breach (or re-breach after recovery) — fire once and hold.
            self.state.kill_switch_active = True
            self.state.paused_until_tick = self.state.current_tick + KILL_SWITCH_PAUSE_TICKS
            log.warning(
                "risk.kill_switch",
                drawdown=str(dd),
                paused_for_ticks=KILL_SWITCH_PAUSE_TICKS,
                resume_at_mult=str(SIZE_MULT_RECOVERY),
                note="after pause resumes at 0.10x recovery mode, not frozen",
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
            # Kill switch active — only exits are permitted
            sells = [i for i in intents if i.side == "SELL"]
            if sells:
                log.info("risk.paused.allow_sells_only", count=len(sells))
            return sells

        size_mult = self._size_multiplier(equity)
        if size_mult < SIZE_MULT_FULL:
            log.info(
                "risk.drawdown_scaling",
                drawdown=str(self.drawdown(equity)),
                multiplier=str(size_mult),
            )

        allowed: list[OrderIntent] = []
        for intent in intents:

            # --- SELLs always pass (we never block an exit)
            if intent.side == "SELL":
                if intent.quantity > MAX_ORDER_QUANTITY:
                    log.info(
                        "risk.cap.quantity.sell_clamped",
                        ticker=intent.ticker,
                        original=intent.quantity,
                        clamped=MAX_ORDER_QUANTITY,
                    )
                    intent = OrderIntent(
                        side=intent.side, ticker=intent.ticker,
                        quantity=MAX_ORDER_QUANTITY, order_type=intent.order_type,
                        limit_price=intent.limit_price, reason=intent.reason,
                        entry_atr=intent.entry_atr,
                    )
                if self._under_rate_limit():
                    allowed.append(intent)
                    self._record_order()
                else:
                    log.warning("risk.rate_limit.exit_dropped", ticker=intent.ticker)
                continue

            # --- BUYs: apply all caps then drawdown scaling

            if intent.quantity > MAX_ORDER_QUANTITY:
                log.info(
                    "risk.cap.quantity.buy_clamped",
                    ticker=intent.ticker,
                    original=intent.quantity,
                    clamped=MAX_ORDER_QUANTITY,
                )
                intent = OrderIntent(
                    side=intent.side, ticker=intent.ticker,
                    quantity=MAX_ORDER_QUANTITY, order_type=intent.order_type,
                    limit_price=intent.limit_price, reason=intent.reason,
                    entry_atr=intent.entry_atr,
                )

            if not self._under_rate_limit():
                log.warning("risk.rate_limit.entry_dropped", ticker=intent.ticker)
                continue

            price = intent.limit_price or Decimal(
                str(market.get(intent.ticker).price if market.get(intent.ticker) else 0)
            )
            notional = price * intent.quantity

            # Order notional cap
            if notional > equity * MAX_ORDER_NOTIONAL:
                log.warning(
                    "risk.cap.order_notional",
                    ticker=intent.ticker,
                    notional=str(notional),
                    cap=str(equity * MAX_ORDER_NOTIONAL),
                )
                continue

            # Per-ticker position cap
            existing    = portfolio.positions.get(intent.ticker)
            new_qty     = (existing.quantity if existing else 0) + intent.quantity
            new_pos_val = price * new_qty
            if new_pos_val > equity * MAX_POSITION_FRACTION:
                log.warning(
                    "risk.cap.position",
                    ticker=intent.ticker,
                    new_value=str(new_pos_val),
                    cap=str(equity * MAX_POSITION_FRACTION),
                )
                continue

            # Per-sector exposure cap
            # Prevents over-concentration during SECTOR_BOOM events where
            # the top-N momentum stocks all happen to be in the same sector.
            sector_exp = self._sector_exposure(intent.ticker, portfolio, market)
            if sector_exp + notional > equity * MAX_SECTOR_EXPOSURE:
                log.info(
                    "risk.cap.sector_exposure",
                    ticker=intent.ticker,
                    sector_exposure=str(sector_exp),
                    cap=str(equity * MAX_SECTOR_EXPOSURE),
                )
                continue

            # Total exposure cap
            current_exposure = sum(
                Decimal(str(market.get(t).price if market.get(t) else 0)) * p.quantity
                for t, p in portfolio.positions.items()
            )
            if current_exposure + notional > equity * MAX_TOTAL_EXPOSURE:
                log.info("risk.cap.total_exposure", ticker=intent.ticker)
                continue

            # Drawdown-proportional quantity scaling
            if size_mult <= 0:
                continue  # should be caught by trading_paused, but be safe
            if size_mult < SIZE_MULT_FULL:
                scaled_qty = max(1, int(intent.quantity * size_mult))
                intent = OrderIntent(
                    side=intent.side,
                    ticker=intent.ticker,
                    quantity=scaled_qty,
                    order_type=intent.order_type,
                    limit_price=intent.limit_price,
                    reason=intent.reason,
                    entry_atr=intent.entry_atr,
                )

            allowed.append(intent)
            self._record_order()

        return allowed

    # ----------------------------------------------------------------- helpers

    def _sector_exposure(
        self,
        ticker: str,
        portfolio: PortfolioView,
        market: MarketStore,
    ) -> Decimal:
        """Current notional held in the same sector as ticker."""
        state  = market.get(ticker)
        sector = (state.sector or "").strip().lower() if state else ""
        if not sector:
            return Decimal("0")
        total = Decimal("0")
        for t, pos in portfolio.positions.items():
            ts = market.get(t)
            if ts and (ts.sector or "").strip().lower() == sector:
                price = Decimal(str(ts.price)) if ts.price > 0 else pos.avg_cost
                total += price * pos.quantity
        return total

    def _under_rate_limit(self) -> bool:
        cutoff = time.monotonic() - 60.0
        while self.state.order_timestamps and self.state.order_timestamps[0] < cutoff:
            self.state.order_timestamps.popleft()
        return len(self.state.order_timestamps) < ORDERS_PER_MINUTE

    def _record_order(self) -> None:
        self.state.order_timestamps.append(time.monotonic())
