"""
Decision engine.

Inputs:  current MarketStore + current Portfolio + cash balance
Outputs: a list of OrderIntent objects (BUY / SELL with quantity)

The strategy is the "concentrated aggressive momentum" plan from the docx:
  * scan all tickers, score by momentum_score
  * keep the top 5 above a dynamic threshold (the median of all positive scores)
  * for each held position, manage a trailing stop and a hard stop
  * for each new top-5 ticker we do NOT yet hold, emit a BUY intent sized
    to 15% of current equity (Day 2 entry rule; pyramid lands Day 3)

No knobs are exposed. Everything is hard-coded with sensible defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterable

import numpy as np

from . import indicators
from .logging_setup import get_logger
from .market import MarketStore, TickerState

log = get_logger(__name__)


# --------------------------------------------------------------------------- types

@dataclass
class OrderIntent:
    """A desired action. The executor turns this into a real /orders call."""
    side: str               # "BUY" | "SELL"
    ticker: str
    quantity: int
    order_type: str         # "LIMIT" | "MARKET"
    limit_price: Decimal | None = None
    reason: str = ""        # for logging: "entry", "trailing_stop", "hard_stop", "exit_watchlist"


@dataclass
class Position:
    ticker: str
    quantity: int
    avg_cost: Decimal
    peak_price: Decimal     # highest price seen since entry — drives trailing stop


@dataclass
class PortfolioView:
    cash: Decimal
    positions: dict[str, Position] = field(default_factory=dict)

    def equity(self, market: MarketStore) -> Decimal:
        total = self.cash
        for t, pos in self.positions.items():
            state = market.get(t)
            price = Decimal(str(state.price)) if state else pos.avg_cost
            total += price * pos.quantity
        return total


# ---------------------------------------------------------------------- constants

# Hard-coded defaults — the bot self-tunes nothing the user touches.
WATCHLIST_SIZE = 5
ENTRY_FRACTION = Decimal("0.15")     # 15% of equity on first entry
PYRAMID_FRACTION = Decimal("0.10")   # 10% of equity on each add-on
PYRAMID_TRIGGER_PCT = Decimal("0.02")  # price must be >=2% above avg cost
ATR_STOP_MULTIPLIER = 2.0            # trailing stop = peak - 2.0 * ATR
HARD_STOP_PCT = Decimal("-0.03")     # -3% from average cost
MIN_PRICES_FOR_SCORING = 31          # need ~30 closes for ATR/breakout/volume_surge
MIN_TICKERS_FOR_ENTRY = 5            # don't try to enter before we have 5+ candidates


# ------------------------------------------------------------------- core function

class Strategy:
    """One strategy, executed well. Stateless between ticks except for caches."""

    def __init__(self) -> None:
        # Caches the last computed score per ticker for debugging / logging
        self._last_scores: dict[str, float] = {}

    def decide(
        self,
        market: MarketStore,
        portfolio: PortfolioView,
    ) -> list[OrderIntent]:
        """Run one tick of the decision pipeline. Returns intents to act on."""
        intents: list[OrderIntent] = []

        # 1. Score everything that has enough data.
        scores = self._score_all(market)
        self._last_scores = scores

        # 2. Pick the top-N watchlist among positive scores.
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        positive = [(t, s) for t, s in ranked if s > 0]
        watchlist = {t for t, _ in positive[:WATCHLIST_SIZE]}

        # 3. Manage existing positions first (exits take priority over entries).
        for ticker, pos in list(portfolio.positions.items()):
            exit_intent = self._exit_decision(ticker, pos, market, in_watchlist=ticker in watchlist)
            if exit_intent:
                intents.append(exit_intent)

        # 4. For each watchlist ticker we don't already hold, emit an entry.
        if len(scores) >= MIN_TICKERS_FOR_ENTRY:
            equity = portfolio.equity(market)
            for ticker in watchlist:
                if ticker in portfolio.positions:
                    continue
                state = market.get(ticker)
                if state is None or state.price <= 0:
                    continue
                entry = self._entry_intent(state, equity, portfolio.cash)
                if entry is not None:
                    intents.append(entry)

            # 5. Pyramid into existing winners that are still scoring high.
            pyramid_intents = self.propose_pyramid(market, portfolio, watchlist)
            intents.extend(pyramid_intents)

        return intents

    def propose_pyramid(
        self,
        market: MarketStore,
        portfolio: PortfolioView,
        watchlist: set[str],
    ) -> list[OrderIntent]:
        """
        Add to winners.

        For each currently-held ticker that is still in the watchlist AND has
        rallied at least PYRAMID_TRIGGER_PCT above its average cost, emit a
        smaller BUY intent (PYRAMID_FRACTION of current equity). The 25%
        per-ticker risk cap will reject the add-on if the position is already
        large enough — that's the intended brake on over-pyramiding.
        """
        intents: list[OrderIntent] = []
        equity = portfolio.equity(market)
        cash = portfolio.cash
        for ticker, pos in portfolio.positions.items():
            if ticker not in watchlist:
                continue
            state = market.get(ticker)
            if state is None or state.price <= 0:
                continue
            price = Decimal(str(state.price))
            trigger = pos.avg_cost * (Decimal("1") + PYRAMID_TRIGGER_PCT)
            if price < trigger:
                continue
            notional = min(equity * PYRAMID_FRACTION, cash)
            if notional <= 0:
                continue
            qty = int(notional / price)
            if qty <= 0:
                continue
            limit_price = (price * Decimal("1.001")).quantize(Decimal("0.01"))
            intents.append(OrderIntent(
                side="BUY",
                ticker=ticker,
                quantity=qty,
                order_type="LIMIT",
                limit_price=limit_price,
                reason="pyramid",
            ))
            # Note: we don't decrement `cash` here because the risk gate
            # rechecks total exposure / cash anyway and we want each pyramid
            # decision to be independent.
        return intents

    # ----------------------------------------------------------------- internals

    def _score_all(self, market: MarketStore) -> dict[str, float]:
        scores: dict[str, float] = {}
        for ticker in market.tickers():
            state = market.get(ticker)
            if state is None or len(state.recent) < MIN_PRICES_FOR_SCORING:
                continue
            closes = np.array(state.recent, dtype=np.float64)
            # Real per-tick volume history (Track A). When the history is
            # shorter than the price history (e.g. the ticker was added late),
            # pad the front with the earliest known volume so the array
            # lengths match.
            if len(state.volumes) == len(state.recent):
                volumes = np.array(state.volumes, dtype=np.float64)
            else:
                pad = state.volumes[0] if state.volumes else state.volume
                volumes = np.array(
                    [pad] * (len(state.recent) - len(state.volumes))
                    + list(state.volumes),
                    dtype=np.float64,
                )
            score = indicators.momentum_score(closes, volumes)
            if score is not None:
                scores[ticker] = score
        return scores

    def _entry_intent(
        self,
        state: TickerState,
        equity: Decimal,
        cash: Decimal,
    ) -> OrderIntent | None:
        price = Decimal(str(state.price))
        notional = equity * ENTRY_FRACTION
        # Don't try to spend more cash than we have.
        notional = min(notional, cash)
        if notional <= 0:
            return None
        qty = int(notional / price)
        if qty <= 0:
            return None
        # LIMIT one tick above current price — we want fills, not bargains.
        limit_price = (price * Decimal("1.001")).quantize(Decimal("0.01"))
        return OrderIntent(
            side="BUY",
            ticker=state.ticker,
            quantity=qty,
            order_type="LIMIT",
            limit_price=limit_price,
            reason="entry",
        )

    def _exit_decision(
        self,
        ticker: str,
        pos: Position,
        market: MarketStore,
        *,
        in_watchlist: bool,
    ) -> OrderIntent | None:
        state = market.get(ticker)
        if state is None or state.price <= 0:
            return None
        price = Decimal(str(state.price))

        # Update peak as we go (caller persists Position objects across ticks).
        if price > pos.peak_price:
            pos.peak_price = price

        # Hard stop: -3% from average cost.
        hard_stop = pos.avg_cost * (Decimal("1") + HARD_STOP_PCT)
        if price <= hard_stop:
            return OrderIntent(
                side="SELL", ticker=ticker, quantity=pos.quantity,
                order_type="MARKET", reason="hard_stop",
            )

        # Trailing stop: peak - 2.0 * ATR.
        closes = np.array(state.recent, dtype=np.float64)
        atr = indicators.atr_proxy(closes, length=14)
        if atr is not None and atr > 0:
            trailing = pos.peak_price - Decimal(str(ATR_STOP_MULTIPLIER * atr))
            if price <= trailing:
                return OrderIntent(
                    side="SELL", ticker=ticker, quantity=pos.quantity,
                    order_type="MARKET", reason="trailing_stop",
                )

        # Universe rotation: if a held ticker drops out of the watchlist, exit it.
        # (Frees capital for the new top-N.)
        if not in_watchlist:
            return OrderIntent(
                side="SELL", ticker=ticker, quantity=pos.quantity,
                order_type="MARKET", reason="exit_watchlist",
            )

        return None  # hold
