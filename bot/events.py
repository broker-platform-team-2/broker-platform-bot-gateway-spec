"""
Event reactor — v2.

Translates exchange MARKET_EVENT messages into OrderIntents — fast.

Event payloads (from the exchange WS server Go struct):
  {
    "event_id":      "<uuid>",
    "event_type":    "BULL_RUN" | "BEAR_CRASH" | "SECTOR_BOOM"
                   | "SECTOR_SLUMP" | "STOCK_SHOCK",
    "scope":         "MARKET" | "SECTOR" | "STOCK",
    "target":        "<sector-name-or-ticker>" | "",
    "magnitude":     <float>,        ← price multiplier: 1.1 = +10% effect
    "duration_ticks": <int>,         ← how many sim ticks the effect lasts
    "headline":      "<string>",
    "market_time":   "<iso>"
  }

How magnitude maps to the simulation formula:
    eventComponent = randomWalk × (magnitude - 1.0) × eventWeight
  * magnitude = 1.0  → no event effect
  * magnitude = 1.10 → +10% amplification of random walk (bullish)
  * magnitude = 0.90 → -10% amplification (bearish)

Changes from v1:
  * magnitude-aware position sizing: bigger events deploy more equity
  * duration_ticks returned to app.py so it can schedule auto-exits
  * same-event debounce: ignores re-triggers within DEBOUNCE_TICKS
  * current_tick passed in so debounce works across the bot lifecycle
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

# Base fraction of equity per event-driven entry before magnitude scaling.
EVENT_ENTRY_BASE = Decimal("0.10")

# Magnitude scaling: fraction = clamp(BASE × impact × 10, 0.05, 0.15)
#   impact = abs(magnitude - 1.0)
#   BULL_RUN magnitude=1.10 → impact=0.10 → fraction = 0.10 × 0.10 × 10 = 0.10
#   BULL_RUN magnitude=1.50 → impact=0.50 → fraction capped at 0.15
#   BULL_RUN magnitude=1.02 → impact=0.02 → fraction floored at 0.05
EVENT_ENTRY_MIN  = Decimal("0.05")
EVENT_ENTRY_MAX  = Decimal("0.15")

# How many tickers to chase per event (same as strategy watchlist)
EVENT_TOP_N = WATCHLIST_SIZE

# Ignore a second event of the same (scope, event_type, target) if it arrives
# within this many price-update counter ticks of the first.
# With 8 tickers at 1 tick/sec: ~8 counter increments/sec → 50 ticks ≈ 6 seconds.
DEBOUNCE_TICKS = 50


# --------------------------------------------------------------- normalisation

def _norm(payload: dict[str, Any], key_snake: str, key_camel: str) -> Any:
    return payload.get(key_snake) or payload.get(key_camel)


def _magnitude_fraction(magnitude: float) -> Decimal:
    """Convert simulation magnitude to an equity fraction for event entries.

    Larger magnitude → bigger price effect → deploy more capital.
    """
    impact = abs(magnitude - 1.0)
    raw = float(EVENT_ENTRY_BASE) * impact * 10.0
    clamped = max(float(EVENT_ENTRY_MIN), min(float(EVENT_ENTRY_MAX), raw))
    return Decimal(str(round(clamped, 4)))


# ---------------------------------------------------------------------- reactor

class EventReactor:
    """Event-to-intent translator. Called once per MARKET_EVENT message."""

    def __init__(self, strategy: Strategy) -> None:
        # We read the strategy's cached score map so we don't recompute on
        # every event; the strategy refreshes it every decision cycle.
        self._strategy = strategy
        # Debounce: track the last counter-tick at which each event key fired.
        self._last_event_tick: dict[str, int] = {}

    def react(
        self,
        payload: dict[str, Any],
        market: MarketStore,
        portfolio: PortfolioView,
        current_tick: int = 0,
    ) -> tuple[list[OrderIntent], int]:
        """Process one MARKET_EVENT.

        Returns (intents, duration_ticks) so app.py can schedule auto-exits
        for positions opened in response to this event.
        duration_ticks == 0 means no auto-exit needed (bearish / shock events
        produce sells immediately, not timed holds).
        """
        event_type = (_norm(payload, "event_type", "eventType") or "").upper()
        scope      = (_norm(payload, "scope", "scope") or "").upper()
        target     = payload.get("target") or ""
        magnitude  = float(payload.get("magnitude") or 1.0)
        duration   = int(payload.get("duration_ticks") or 0)

        log.info(
            "event.received",
            event_type=event_type,
            scope=scope,
            target=target,
            magnitude=magnitude,
            duration_ticks=duration,
            headline=payload.get("headline"),
        )

        # --- debounce: same event type + scope + target within DEBOUNCE_TICKS
        dedup_key = f"{event_type}:{scope}:{target}"
        last_tick = self._last_event_tick.get(dedup_key, -(DEBOUNCE_TICKS + 1))
        if current_tick - last_tick < DEBOUNCE_TICKS:
            log.info(
                "event.debounced",
                key=dedup_key,
                ticks_since_last=current_tick - last_tick,
            )
            return [], 0
        self._last_event_tick[dedup_key] = current_tick

        # --- bullish: chase top-momentum tickers in the affected universe
        if event_type in ("BULL_RUN", "SECTOR_BOOM"):
            tickers  = self._affected_universe(market, scope, target)
            fraction = _magnitude_fraction(magnitude)
            intents  = self._bull_intents(tickers, market, portfolio, fraction)
            log.info(
                "event.bull_entries",
                event_type=event_type,
                fraction=str(fraction),
                magnitude=magnitude,
                duration_ticks=duration,
                entries=len(intents),
            )
            return intents, duration

        # --- bearish: immediately flatten everything in the affected universe
        if event_type in ("BEAR_CRASH", "SECTOR_SLUMP"):
            tickers = self._affected_universe(market, scope, target)
            intents = self._flatten_intents(
                tickers, portfolio, reason=f"event_{event_type.lower()}"
            )
            log.info(
                "event.bear_flatten",
                event_type=event_type,
                magnitude=magnitude,
                sells=len(intents),
            )
            return intents, 0  # no timed exits — we already sold

        # --- stock shock: exit the named ticker if held; direction unknown
        if event_type == "STOCK_SHOCK":
            if target and target in portfolio.positions:
                intents = self._flatten_intents(
                    [target], portfolio, reason="event_stock_shock"
                )
                return intents, 0
            return [], 0

        log.warning("event.unknown_type", event_type=event_type)
        return [], 0

    # ----------------------------------------------------------------- helpers

    def _affected_universe(
        self,
        market: MarketStore,
        scope: str,
        target: str,
    ) -> list[str]:
        if scope == "MARKET":
            return market.tickers()
        if scope == "SECTOR" and target:
            return market.tickers_in_sector(target)
        if scope == "STOCK" and target:
            return [target]
        # Unknown scope — conservative fallback to market-wide
        return market.tickers()

    def _bull_intents(
        self,
        tickers: list[str],
        market: MarketStore,
        portfolio: PortfolioView,
        fraction: Decimal,
    ) -> list[OrderIntent]:
        """Buy the top-N highest-scoring tickers in the affected universe.

        If scores are not yet available (early in the session), we still act —
        the event's magnitude matters more than perfect ranking at this point.
        """
        scores = self._strategy._last_scores  # noqa: SLF001 — same-package read
        ranked = sorted(
            ((t, scores.get(t, 0.0)) for t in tickers),
            key=lambda kv: kv[1],
            reverse=True,
        )
        top = [t for t, _ in ranked[:EVENT_TOP_N]]

        equity  = portfolio.equity(market)
        cash    = portfolio.cash
        intents: list[OrderIntent] = []
        for ticker in top:
            state = market.get(ticker)
            if state is None or state.price <= 0:
                continue
            notional = min(equity * fraction, cash)
            if notional <= 0:
                continue
            price = Decimal(str(state.price))
            qty   = int(notional / price)
            if qty <= 0:
                continue
            # MARKET order — event speed beats price precision
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