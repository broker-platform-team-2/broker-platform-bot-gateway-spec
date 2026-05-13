"""
Event reactor.

Translates exchange MARKET_EVENT messages into OrderIntents — fast.

Event payloads (from the exchange spec):

  {
    "event_type": "BULL_RUN" | "BEAR_CRASH" | "SECTOR_BOOM"
                 | "SECTOR_SLUMP" | "STOCK_SHOCK",
    "scope":      "MARKET" | "SECTOR" | "STOCK",
    "target":     "<sector-name-or-ticker>" | null,
    "magnitude":  <float>,
    "headline":   "<headline string>",
    "duration_ticks": <int>,
    "market_time": "<iso>"
  }

The plan: event speed beats event certainty. We act on the FIRST message of
its kind without waiting for confirmation. False positives cost us a few
percent; missed events cost us the competition.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from .logging_setup import get_logger
from .market import MarketStore
from .strategy import (
    OrderIntent,
    PortfolioView,
    Strategy,
    WATCHLIST_SIZE,
)

log = get_logger(__name__)


# --------------------------------------------------------------------- constants

# How much equity to deploy per event-driven entry. Smaller than first-entry
# (15%) because we may pile into several event-driven names in one go.
EVENT_ENTRY_FRACTION = Decimal("0.10")

# How many tickers to chase on a BULL_RUN / SECTOR_BOOM (matches watchlist).
EVENT_TOP_N = WATCHLIST_SIZE


# --------------------------------------------------------------- normalisation

def _norm(payload: dict[str, Any], key_snake: str, key_camel: str) -> Any:
    return payload.get(key_snake) or payload.get(key_camel)


# ---------------------------------------------------------------------- reactor

class EventReactor:
    """Stateless event-to-intent translator. Called once per MARKET_EVENT."""

    def __init__(self, strategy: Strategy) -> None:
        # We read the strategy's cached score map so we don't recompute on
        # every event; the strategy refreshes it every decision cycle.
        self._strategy = strategy

    def react(
        self,
        payload: dict[str, Any],
        market: MarketStore,
        portfolio: PortfolioView,
    ) -> list[OrderIntent]:
        event_type = _norm(payload, "event_type", "eventType") or ""
        scope = _norm(payload, "scope", "scope") or ""
        target = payload.get("target")

        event_type = event_type.upper()
        scope = scope.upper()

        log.info(
            "event.received",
            event_type=event_type,
            scope=scope,
            target=target,
            headline=payload.get("headline"),
        )

        # Bullish: chase top-momentum tickers in the affected universe.
        if event_type in ("BULL_RUN", "SECTOR_BOOM"):
            tickers = self._affected_universe(market, scope, target)
            return self._bull_intents(tickers, market, portfolio)

        # Bearish: flatten everything we hold inside the affected universe.
        if event_type in ("BEAR_CRASH", "SECTOR_SLUMP"):
            tickers = self._affected_universe(market, scope, target)
            return self._flatten_intents(tickers, portfolio, reason=f"event_{event_type.lower()}")

        # Stock shock: just exit the named ticker if we hold it. No new entries —
        # shocks can go either direction, and the held position is what's at risk.
        if event_type == "STOCK_SHOCK":
            if isinstance(target, str) and target in portfolio.positions:
                return self._flatten_intents([target], portfolio, reason="event_stock_shock")
            return []

        log.warning("event.unknown_type", event_type=event_type)
        return []

    # ----------------------------------------------------------------- helpers

    def _affected_universe(
        self,
        market: MarketStore,
        scope: str,
        target: Any,
    ) -> list[str]:
        if scope == "MARKET":
            return market.tickers()
        if scope == "SECTOR" and isinstance(target, str):
            return market.tickers_in_sector(target)
        if scope == "STOCK" and isinstance(target, str):
            return [target]
        # Unknown scope: be conservative and assume market-wide.
        return market.tickers()

    def _bull_intents(
        self,
        tickers: list[str],
        market: MarketStore,
        portfolio: PortfolioView,
    ) -> list[OrderIntent]:
        # Pick the top-N by cached score within the affected universe. If
        # the score cache is empty (event before the strategy has run a cycle),
        # fall back to all tickers in the universe.
        scores = self._strategy._last_scores  # noqa: SLF001 — same-package read
        ranked = sorted(
            ((t, scores.get(t, 0.0)) for t in tickers),
            key=lambda kv: kv[1],
            reverse=True,
        )
        # Take the top-N. If we have no score data yet, ranked is still ordered
        # arbitrarily (all zeros) — that's OK, we still want to act on speed.
        top = [t for t, _ in ranked[:EVENT_TOP_N]]

        equity = portfolio.equity(market)
        cash = portfolio.cash
        intents: list[OrderIntent] = []
        for ticker in top:
            state = market.get(ticker)
            if state is None or state.price <= 0:
                continue
            # Don't re-enter what we already hold heavily; the risk gate's
            # 25% per-ticker cap will block oversized adds anyway.
            notional = min(equity * EVENT_ENTRY_FRACTION, cash)
            if notional <= 0:
                continue
            price = Decimal(str(state.price))
            qty = int(notional / price)
            if qty <= 0:
                continue
            # Use MARKET on event-driven entries — we want fills NOW, not later.
            intents.append(OrderIntent(
                side="BUY",
                ticker=ticker,
                quantity=qty,
                order_type="MARKET",
                reason="event_bull",
            ))
        return intents

    def _flatten_intents(
        self,
        tickers: list[str],
        portfolio: PortfolioView,
        *,
        reason: str,
    ) -> list[OrderIntent]:
        intents: list[OrderIntent] = []
        for ticker in tickers:
            pos = portfolio.positions.get(ticker)
            if not pos or pos.quantity <= 0:
                continue
            intents.append(OrderIntent(
                side="SELL",
                ticker=ticker,
                quantity=pos.quantity,
                order_type="MARKET",
                reason=reason,
            ))
        return intents
