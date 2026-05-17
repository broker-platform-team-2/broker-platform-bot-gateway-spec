"""
Decision engine — v2.

Designed around the price-simulation engine's formula:
    new_price = current
              + (volatility × random_walk) + trendBias      ← noise + drift
              + (pressureRatio × current × momentum)         ← order pressure
              + randomWalk × (magnitude - 1) × eventWeight   ← event amplifier

Since the Go WS server strips volatility/trendBias/momentum/eventWeight from
PRICE_UPDATE, the strategy approximates them:
  - ATR proxy    ≈  volatility × current_price  (tick-to-tick noise level)
  - volume_surge ≈  order pressure              (high volume = buy/sell imbalance)
  - rate_of_change ≈ trendBias effect           (persistent drift)
  - MARKET_EVENT magnitude + duration_ticks     (known from exchange spec)

Changes from v1:
  * exit_watchlist replaced by EXIT_SCORE_THRESHOLD (score < -0.01 → sell)
  * ATR-based hard stop anchored at entry (entry_atr stored on Position)
  * Trailing stop delayed until +5% profit, widened from 2× to 3× ATR
  * Watchlist expanded to 8; entry sizes tiered by rank (top-3 vs 4-8)
  * Fast-path warmup entries via change_pct for tickers with < 31 prices
  * Sector-relative alpha blended into composite score (0.15 weight)
  * Order-book imbalance signal (0.10 weight, reflects order pressure)
  * Spread filter: skip tickers with bid-ask spread > 0.5%
  * entry_atr threaded through OrderIntent so app.py stamps it on Position
  * Revival mechanism: after FLAT_REVIVAL_CYCLES flat decision cycles, make
    one minimum entry in best ticker to prevent permanent freeze (path 2)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np

from . import indicators
from .logging_setup import get_logger
from .market import MarketStore, TickerState
from .orderbook import OrderBook

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
    # ATR at the moment of entry — used for the hard stop on this position.
    # Passed through to app.py which stamps it onto the Position record.
    entry_atr: Decimal = field(default_factory=lambda: Decimal("0"))


@dataclass
class Position:
    ticker: str
    quantity: int
    avg_cost: Decimal
    peak_price: Decimal
    # ATR recorded at entry. Zero means "use fallback fixed-pct stop".
    entry_atr: Decimal = field(default_factory=lambda: Decimal("0"))


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

WATCHLIST_SIZE = 8

# Entry sizing — tiered by conviction rank
ENTRY_FRACTION_TIER1 = Decimal("0.12")    # rank 1-3: high conviction
ENTRY_FRACTION_TIER2 = Decimal("0.08")    # rank 4-8: lower conviction
ENTRY_FRACTION_FAST  = Decimal("0.04")    # warmup fast-path: half of tier2

PYRAMID_FRACTION     = Decimal("0.10")
PYRAMID_TRIGGER_PCT  = Decimal("0.02")

# Exit thresholds
EXIT_SCORE_THRESHOLD   = -0.01             # sell only when momentum clearly negative
HARD_STOP_ATR_MULT     = Decimal("2.5")    # hard stop = avg_cost - 2.5 × entry_atr
HARD_STOP_PCT_FALLBACK = Decimal("-0.03")  # fallback when entry_atr == 0
TRAILING_ATR_MULT      = 3.0              # wider trailing: 3× current ATR
TRAILING_START_PCT     = Decimal("0.05")   # trailing only kicks in after +5% profit

# Scoring
MIN_PRICES_FOR_SCORING      = 31  # 31 ticks needed for volume_surge(length=30)
MIN_PRICES_FOR_FAST_SCORING =  5  # change_pct fast path
MIN_TICKERS_FOR_ENTRY       =  5  # need >=5 scored tickers before entering

# Signal blend weights (added on top of base momentum_score)
SECTOR_ALPHA_WEIGHT  = 0.15   # reward outperforming own sector
OB_IMBALANCE_WEIGHT  = 0.10   # order-book buy/sell pressure (approx simulation order_pressure)

# Liquidity quality gate
SPREAD_FILTER_MAX_PCT = Decimal("0.005")   # skip tickers with spread > 0.5%

# Revival: if the bot has been completely flat (no positions, no entries) for
# this many consecutive decision cycles it makes one tiny entry in the best
# available ticker even when all scores are <= 0.  This is a heartbeat — not
# aggression — to prevent the second bot-freeze path where every score flips
# negative and the positive watchlist stays empty indefinitely.
FLAT_REVIVAL_CYCLES = 20   # ~30 s at 1 tick/s, DECISION_INTERVAL = 12 ticks


# ------------------------------------------------------------------- strategy

class Strategy:
    """Concentrated aggressive momentum — designed around the simulation model."""

    def __init__(self) -> None:
        self._last_scores: dict[str, float] = {}
        self._flat_decision_cycles: int = 0   # consecutive cycles with no position & no entry

    # ----------------------------------------------------------------- public

    def decide(
        self,
        market: MarketStore,
        portfolio: PortfolioView,
        orderbook: OrderBook | None = None,
    ) -> list[OrderIntent]:
        intents: list[OrderIntent] = []

        scores = self._score_all(market, orderbook=orderbook)
        self._last_scores = scores

        ranked   = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        positive = [(t, s) for t, s in ranked if s > 0]
        watchlist = {t for t, _ in positive[:WATCHLIST_SIZE]}

        # --- exits: every held position, regardless of watchlist rank
        for ticker, pos in list(portfolio.positions.items()):
            exit_intent = self._exit_decision(ticker, pos, market)
            if exit_intent:
                intents.append(exit_intent)

        # --- entries + pyramiding (only once we have enough scored tickers)
        if len(scores) >= MIN_TICKERS_FOR_ENTRY:
            equity = portfolio.equity(market)
            for rank, (ticker, _) in enumerate(positive[:WATCHLIST_SIZE]):
                if ticker in portfolio.positions:
                    continue
                state = market.get(ticker)
                if state is None or state.price <= 0:
                    continue
                if orderbook and not self._spread_ok(ticker, orderbook):
                    log.debug("strategy.spread_filter.skipped", ticker=ticker)
                    continue
                fraction = ENTRY_FRACTION_TIER1 if rank < 3 else ENTRY_FRACTION_TIER2
                entry = self._entry_intent(state, equity, portfolio.cash, fraction)
                if entry is not None:
                    intents.append(entry)

            intents.extend(self.propose_pyramid(market, portfolio, watchlist))

        # --- fast-path warmup: act before MIN_TICKERS_FOR_ENTRY are scored
        else:
            intents.extend(
                self._fast_path_entries(market, portfolio, scores, orderbook)
            )

        # ---- Revival: prevent the "all scores negative -> permanently flat" freeze
        #
        # Path 1 (kill switch) is handled in risk.py (0.10x after 30-tick pause).
        # Path 2: all tickers score <= 0 -> positive = [] -> zero entries forever.
        # After FLAT_REVIVAL_CYCLES consecutive flat decision cycles we force one
        # minimum entry in the best-ranked ticker regardless of score sign.
        # This still passes through every risk cap — if cash or equity caps block
        # it the bot waits another FLAT_REVIVAL_CYCLES before trying again.
        has_positions = bool(portfolio.positions)
        has_entries   = any(i.side == "BUY" for i in intents)

        if not has_positions and not has_entries:
            self._flat_decision_cycles += 1
            log.debug(
                "strategy.flat_counter",
                flat_cycles=self._flat_decision_cycles,
                until_revival=max(0, FLAT_REVIVAL_CYCLES - self._flat_decision_cycles),
            )
            if self._flat_decision_cycles >= FLAT_REVIVAL_CYCLES:
                self._flat_decision_cycles = 0
                if ranked:  # ranked is the full score list, not just positive
                    top_ticker, top_score = ranked[0]
                    state = market.get(top_ticker)
                    if state and state.price > 0:
                        equity = portfolio.equity(market)
                        revival = self._entry_intent(
                            state, equity, portfolio.cash, ENTRY_FRACTION_FAST
                        )
                        if revival is not None:
                            revival.reason = "entry_revival"
                            intents.append(revival)
                            log.info(
                                "strategy.revival_entry",
                                ticker=top_ticker,
                                score=round(top_score, 4),
                                note="flat too long — minimum entry to stay alive",
                            )
        else:
            self._flat_decision_cycles = 0

        return intents

    def propose_pyramid(
        self,
        market: MarketStore,
        portfolio: PortfolioView,
        watchlist: set[str],
    ) -> list[OrderIntent]:
        """Add to winners still in top-N and up >=2% from avg cost."""
        intents: list[OrderIntent] = []
        equity = portfolio.equity(market)
        cash   = portfolio.cash
        for ticker, pos in portfolio.positions.items():
            if ticker not in watchlist:
                continue
            state = market.get(ticker)
            if state is None or state.price <= 0:
                continue
            price   = Decimal(str(state.price))
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
                entry_atr=self._current_atr(state),
            ))
        return intents

    # ----------------------------------------------------------------- scoring

    def _score_all(
        self,
        market: MarketStore,
        orderbook: OrderBook | None = None,
    ) -> dict[str, float]:
        """Full momentum score + sector alpha + order book imbalance.

        The three components map to simulation mechanics:
          momentum_score  -> measures drift (trendBias) + volume pressure
          OB imbalance    -> directly measures order_pressure direction
          sector alpha    -> isolates stock-specific signal from sector-wide events
        """
        raw: dict[str, float] = {}

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
                    [pad] * (len(state.recent) - len(state.volumes)) + list(state.volumes),
                    dtype=np.float64,
                )
            score = indicators.momentum_score(closes, volumes)
            if score is None:
                continue

            # Order-book imbalance reflects the simulation's order_pressure component.
            # Positive -> more buyers -> price should drift up next tick.
            if orderbook:
                imb = orderbook.imbalance(ticker)
                if imb is not None:
                    score += OB_IMBALANCE_WEIGHT * imb

            raw[ticker] = score

        # Sector-relative alpha.
        # Simulation events hit whole sectors uniformly (SECTOR_BOOM lifts all stocks
        # in that sector). A stock beating its sector average has stock-specific alpha
        # beyond the event component — worth overweighting.
        sector_avgs = self._sector_averages(market, raw)
        final: dict[str, float] = {}
        for ticker, score in raw.items():
            state  = market.get(ticker)
            sector = (state.sector or "").strip() if state else ""
            if sector and sector in sector_avgs:
                alpha = score - sector_avgs[sector]
                score = score + SECTOR_ALPHA_WEIGHT * alpha
            final[ticker] = score

        return final

    def _sector_averages(
        self,
        market: MarketStore,
        scores: dict[str, float],
    ) -> dict[str, float]:
        buckets: dict[str, list[float]] = {}
        for ticker, score in scores.items():
            state  = market.get(ticker)
            sector = (state.sector or "").strip() if state else ""
            if not sector:
                continue
            buckets.setdefault(sector, []).append(score)
        return {s: sum(v) / len(v) for s, v in buckets.items() if v}

    # ------------------------------------------------------------- fast path

    def _fast_path_entries(
        self,
        market: MarketStore,
        portfolio: PortfolioView,
        full_scores: dict[str, float],
        orderbook: OrderBook | None,
    ) -> list[OrderIntent]:
        """Warmup entries before the full scorer has enough data.

        Uses change_pct (cumulative return from daily open) as a proxy.
        change_pct > 0 means price is above the open — consistent with a
        positive trendBias or a recent event still being active.
        Entry size is 1/3 of normal to reflect lower signal quality.
        """
        fast: dict[str, float] = {}
        for ticker in market.tickers():
            if ticker in full_scores:
                continue
            state = market.get(ticker)
            if state is None or len(state.recent) < MIN_PRICES_FOR_FAST_SCORING:
                continue
            if state.price <= 0:
                continue
            # Normalise change_pct (e.g. 3.5%) -> 0.035, comparable scale to
            # momentum_score which is typically +-0.05 in calm markets
            fast[ticker] = state.change_pct / 100.0

        if not fast:
            return []

        equity  = portfolio.equity(market)
        top     = [
            (t, s)
            for t, s in sorted(fast.items(), key=lambda kv: kv[1], reverse=True)
            if s > 0
        ]
        intents: list[OrderIntent] = []
        for ticker, _ in top[:WATCHLIST_SIZE]:
            if ticker in portfolio.positions:
                continue
            state = market.get(ticker)
            if state is None or state.price <= 0:
                continue
            if orderbook and not self._spread_ok(ticker, orderbook):
                continue
            entry = self._entry_intent(state, equity, portfolio.cash, ENTRY_FRACTION_FAST)
            if entry is not None:
                entry.reason = "entry_fast_path"
                intents.append(entry)
        return intents

    # --------------------------------------------------------------- exits

    def _exit_decision(
        self,
        ticker: str,
        pos: Position,
        market: MarketStore,
    ) -> OrderIntent | None:
        state = market.get(ticker)
        if state is None or state.price <= 0:
            return None
        price = Decimal(str(state.price))

        # Track peak for trailing stop
        if price > pos.peak_price:
            pos.peak_price = price

        # 1. Hard stop — ATR-based when we have entry_atr, fixed-pct fallback.
        #    Rationale: simulation random_walk ~= volatility * price per tick.
        #    A drop of 2.5x the entry-tick noise is structural, not random walk.
        if pos.entry_atr > 0:
            hard_stop = pos.avg_cost - HARD_STOP_ATR_MULT * pos.entry_atr
        else:
            hard_stop = pos.avg_cost * (Decimal("1") + HARD_STOP_PCT_FALLBACK)

        if price <= hard_stop:
            return OrderIntent(
                side="SELL",
                ticker=ticker,
                quantity=pos.quantity,
                order_type="MARKET",
                reason="hard_stop",
            )

        # 2. Trailing stop — activates only after position gains +5% from avg cost.
        #    Rationale: let winners run but protect accumulated profit.
        #    Trail distance = TRAILING_ATR_MULT * entry_atr (or 3% of peak if no ATR).
        if price >= pos.avg_cost * (Decimal("1") + TRAILING_START_PCT):
            if pos.entry_atr > 0:
                trail_stop = (
                    pos.peak_price
                    - Decimal(str(TRAILING_ATR_MULT)) * pos.entry_atr
                )
            else:
                trail_stop = pos.peak_price * Decimal("0.97")
            if price <= trail_stop:
                return OrderIntent(
                    side="SELL",
                    ticker=ticker,
                    quantity=pos.quantity,
                    order_type="MARKET",
                    reason="trailing_stop",
                )

        # 3. Score threshold exit — momentum has flipped or dried up.
        score = self._last_scores.get(ticker, 0.0)
        if score < EXIT_SCORE_THRESHOLD:
            return OrderIntent(
                side="SELL",
                ticker=ticker,
                quantity=pos.quantity,
                order_type="MARKET",
                reason="score_exit",
            )

        return None

    # ---------------------------------------------------------------- order helpers

    def _entry_intent(
        self,
        state: TickerState,
        equity: Decimal,
        cash: Decimal,
        fraction: float,
    ) -> OrderIntent | None:
        """Build a BUY OrderIntent for *fraction* of equity, stamped with ATR."""
        notional = min(equity * Decimal(str(fraction)), cash)
        if notional <= 0:
            return None
        price = Decimal(str(state.price))
        qty = int(notional / price)
        if qty <= 0:
            return None
        return OrderIntent(
            side="BUY",
            ticker=state.ticker,
            quantity=qty,
            order_type="MARKET",
            reason="entry",
            entry_atr=self._current_atr(state),
        )

    def _current_atr(self, state: TickerState) -> Decimal:
        """ATR proxy: mean absolute tick-to-tick price change over recent window.

        Maps to simulation's volatility * price component — the tick-level noise
        each stock exhibits.  Used for hard-stop and trailing-stop distances.
        """
        prices = list(state.recent)
        if len(prices) < 2:
            return Decimal("0")
        diffs = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
        atr = sum(diffs) / len(diffs)
        return Decimal(str(round(atr, 6)))

    def _spread_ok(self, ticker: str, orderbook: OrderBook) -> bool:
        """Return True when the bid-ask spread is within the acceptable threshold.

        Wide spreads in the simulation indicate low activity or a market
        disruption tick — we skip entry to avoid paying excessive slippage.
        """
        spread = orderbook.spread_pct(ticker)
        if spread is None:
            return True  # No order-book data yet — don't filter
        return spread <= SPREAD_FILTER_MAX_PCT
