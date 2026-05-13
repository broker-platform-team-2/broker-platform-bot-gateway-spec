"""
Decision engine.

Inputs:  current MarketStore + current Portfolio + cash balance
Outputs: a list of OrderIntent objects (BUY / SELL with quantity)

The strategy is the "concentrated aggressive momentum" plan:
  * scan all tickers, score by momentum_score
  * keep the top 5 with positive score
  * for each held position, manage a trailing stop and a hard stop
  * for each new top-5 ticker we do NOT yet hold, emit a BUY intent sized
    to 15% of current equity
  * pyramid into winners that are still scoring high (Track A)

No knobs are exposed. Everything is hard-coded with sensible defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

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
    reason: str = ""


@dataclass
class Position:
    ticker: str
    quantity: int
    avg_cost: Decimal
    peak_price: Decimal


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

WATCHLIST_SIZE = 5
ENTRY_FRACTION = Decimal("0.15")
PYRAMID_FRACTION = Decimal("0.10")
PYRAMID_TRIGGER_PCT = Decimal("0.02")
ATR_STOP_MULTIPLIER = 2.0
HARD_STOP_PCT = Decimal("-0.03")
MIN_PRICES_FOR_SCORING = 31
MIN_TICKERS_FOR_ENTRY = 5


# ------------------------------------------------------------------- core function

class Strategy:
    """One strategy, executed well. Stateless between ticks except for caches."""

    def __init__(self) -> None:
        self._last_scores: dict[str, float] = {}

    def decide(self, market: MarketStore, portfolio: PortfolioView) -> list[OrderIntent]:
        intents: list[OrderIntent] = []

        scores = self._score_all(market)
        self._last_scores = scores

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        positive = [(t, s) for t, s in ranked if s > 0]
        watchlist = {t for t, _ in positive[:WATCHLIST_SIZE]}

        for ticker, pos in list(portfolio.positions.items()):
            exit_intent = self._exit_decision(ticker, pos, market, in_watchlist=ticker in watchlist)
            if exit_intent:
                intents.append(exit_intent)

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

            pyramid_intents = self.propose_pyramid(market, portfolio, watchlist)
            intents.extend(pyramid_intents)

        return intents

    def propose_pyramid(
        self,
        market: MarketStore,
        portfolio: PortfolioView,
        watchlist: set[str],
    ) -> list[OrderIntent]:
        """Add to winners that are still in the top-N and up >=2% from cost."""
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
        return intents

    # ----------------------------------------------------------------- internals

    def _score_all(self, market: MarketStore) -> dict[str, float]:
        scores: dict[str, float] = {}
        for ticker in market.tickers():
            state = market.get(ticker)
            if state is None or len(state.recent) < MIN_PRICES_FOR_SCORING:
                continue
            closes = np.array(state.recent, dtype=np.float64)
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
        notional = min(equity * ENTRY_FRACTION, cash)
        if notional <= 0:
            return None
        qty = int(notional / price)
        if qty <= 0:
            return None
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

        if price > pos.peak_price:
            pos.peak_price = price

        hard_stop = pos.avg_cost * (Decimal("1") + HARD_STOP_PCT)
        if price <= hard_stop:
            return OrderIntent(
                side="SELL", ticker=ticker, quantity=pos.quantity,
                order_type="MARKET", reason="hard_stop",
            )

        closes = np.array(state.recent, dtype=np.float64)
        atr = indicators.atr_proxy(closes, length=14)
        if atr is not None and atr > 0:
            trailing = pos.peak_price - Decimal(str(ATR_STOP_MULTIPLIER * atr))
            if price <= trailing:
                return OrderIntent(
                    side="SELL", ticker=ticker, quantity=pos.quantity,
                    order_type="MARKET", reason="trailing_stop",
                )

        if not in_watchlist:
            return OrderIntent(
                side="SELL", ticker=ticker, quantity=pos.quantity,
                order_type="MARKET", reason="exit_watchlist",
            )

        return None
