"""
Offline simulation — no broker connection needed.

Runs the full production pipeline against a scripted market scenario:
  MarketStore -> Strategy (momentum + pyramiding)
              -> EventReactor (BULL_RUN / BEAR_CRASH / SECTOR_BOOM / etc.)
              -> RiskGate
              -> simulated fills at market price

The scenario is designed to exercise every feature of the combined Track A + B:
  - Momentum scoring kicks in after ~31 ticks of history
  - Entries into the top-5 movers
  - Pyramiding into winners that keep climbing
  - BULL_RUN event  -> fast market-wide entries
  - SECTOR_BOOM     -> concentrated tech entries
  - STOCK_SHOCK     -> immediate single-stock exit
  - SECTOR_SLUMP    -> flatten all energy positions
  - BEAR_CRASH      -> flatten everything
  - Hard stop on a stock that falls -3%
  - Trailing stop on a stock that peaks then retreats

Output:
  - Prints a live decision log to the console (one line per decision)
  - Saves runs/sim_<timestamp>.jsonl  (readable by `python -m bot replay` and the dashboard)
  - Prints a PnL summary at the end

Run from the project root:
    python scripts/simulate.py
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from decimal import Decimal
from pathlib import Path

# Make sure the package is importable from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.events import EventReactor
from bot.logging_setup import configure as configure_logging
from bot.market import MarketStore
from bot.replay import RunRecorder
from bot.risk import RiskGate
from bot.strategy import OrderIntent, PortfolioView, Position, Strategy

# Suppress INFO/WARNING chatter — simulation prints its own decision log.
configure_logging("ERROR")

# ------------------------------------------------------------------ configuration

SEED = 42
SEED_CASH = Decimal("100_000")
TICKS = 2_000          # total price-update ticks
DECISION_EVERY = 12    # run strategy every N ticks (mirrors app.py)

random.seed(SEED)

# ------------------------------------------------------------------ ticker universe
# (ticker, start_price, drift_per_tick, volatility, sector)
TICKERS = [
    ("NVDA", 480.0,  0.00060,  0.012, "TECH"),     # strong uptrend — will dominate momentum
    ("AAPL", 172.0,  0.00025,  0.007, "TECH"),     # steady climber
    ("MSFT", 415.0,  0.00015,  0.006, "TECH"),     # slow grind up
    ("XOM",   115.0, 0.00020,  0.009, "ENERGY"),   # moderate energy stock
    ("CVX",    96.0, 0.00010,  0.013, "ENERGY"),   # volatile energy
    ("JPM",   195.0, 0.00018,  0.006, "FINANCE"),  # slow finance
    ("GS",    440.0, 0.00022,  0.008, "FINANCE"),  # moderate finance
    ("BAC",    36.0,-0.00040,  0.010, "FINANCE"),  # declining — will hit hard stop
]

# ------------------------------------------------------------------ scheduled events
# (tick, event_payload)
EVENTS = [
    (350,  {"event_type": "BULL_RUN",    "scope": "MARKET", "target": None,
            "magnitude": 1.8, "duration_ticks": 30,
            "headline": "Broad market rally — all sectors surging!"}),

    (700,  {"event_type": "SECTOR_BOOM", "scope": "SECTOR", "target": "TECH",
            "magnitude": 2.1, "duration_ticks": 25,
            "headline": "AI demand surge: Tech sector exploding higher"}),

    (950,  {"event_type": "STOCK_SHOCK", "scope": "STOCK",  "target": "BAC",
            "magnitude": -2.5, "duration_ticks": 10,
            "headline": "BAC: unexpected loan-loss reserve announcement"}),

    (1200, {"event_type": "SECTOR_SLUMP","scope": "SECTOR", "target": "ENERGY",
            "magnitude": -1.6, "duration_ticks": 20,
            "headline": "OPEC+ surprise production increase crushes energy stocks"}),

    (1600, {"event_type": "BEAR_CRASH",  "scope": "MARKET", "target": None,
            "magnitude": -2.0, "duration_ticks": 30,
            "headline": "Flash crash: panic selling across all markets"}),
]

# ------------------------------------------------------------------ price generator

def generate_prices(ticks: int) -> dict[str, list[tuple[float, float]]]:
    """
    Returns {ticker: [(price, volume), ...]} with `ticks` entries each.
    Injects a volume spike whenever a scheduled BULL/BOOM event fires on that
    ticker's sector so the volume_surge indicator sees something real.
    """
    event_ticks_bullish = {e[0] for e in EVENTS if e[1]["event_type"] in ("BULL_RUN", "SECTOR_BOOM")}
    event_ticks_bearish = {e[0] for e in EVENTS if e[1]["event_type"] in ("BEAR_CRASH", "SECTOR_SLUMP")}

    result: dict[str, list[tuple[float, float]]] = {}
    for ticker, start, drift, vol, sector in TICKERS:
        prices: list[tuple[float, float]] = []
        price = start
        for t in range(ticks):
            # Price: GBM
            shock = random.gauss(0, vol)
            # At BEAR_CRASH ticks add a negative spike
            if t in event_ticks_bearish:
                shock -= 0.04
            # At BULL_RUN / SECTOR_BOOM add a positive spike (tech tickers only for boom)
            if t in event_ticks_bullish:
                evt = next(e[1] for e in EVENTS if e[0] == t)
                if evt["scope"] == "MARKET" or evt.get("target") == sector:
                    shock += 0.03
            price = price * math.exp(drift + shock)
            price = max(price, 0.01)

            # Volume: base 1000 + random, spike at event ticks and every ~200 ticks
            base_vol = 1000 + random.randint(-200, 200)
            if t in event_ticks_bullish or t % 200 == 0:
                base_vol = int(base_vol * random.uniform(3.0, 5.0))
            volume = float(base_vol)
            prices.append((round(price, 2), volume))
        result[ticker] = prices
    return result


# ------------------------------------------------------------------ fill simulation

def apply_fill(portfolio: PortfolioView, intent: OrderIntent, price: Decimal) -> None:
    """Simulate an immediate fill at `price`. Modifies portfolio in place."""
    if intent.side == "BUY":
        cost = price * intent.quantity
        if cost > portfolio.cash:
            return
        portfolio.cash -= cost
        pos = portfolio.positions.get(intent.ticker)
        if pos:
            total_qty = pos.quantity + intent.quantity
            new_avg = (pos.avg_cost * pos.quantity + cost) / total_qty
            pos.quantity = total_qty
            pos.avg_cost = new_avg
            pos.peak_price = max(pos.peak_price, price)
        else:
            portfolio.positions[intent.ticker] = Position(
                ticker=intent.ticker,
                quantity=intent.quantity,
                avg_cost=price,
                peak_price=price,
            )
    elif intent.side == "SELL":
        pos = portfolio.positions.get(intent.ticker)
        if pos:
            qty = min(intent.quantity, pos.quantity)
            portfolio.cash += price * qty
            if qty >= pos.quantity:
                del portfolio.positions[intent.ticker]
            else:
                pos.quantity -= qty


# ------------------------------------------------------------------ colour helpers

RESET = "\033[0m"
GREEN = "\033[32m"
RED   = "\033[31m"
CYAN  = "\033[36m"
BOLD  = "\033[1m"
YELLOW = "\033[33m"

def _colour(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


# ------------------------------------------------------------------ main simulation

def run() -> None:
    print(f"\n{BOLD}=== Offline Simulation — Spring Practice 2026 Bot ==={RESET}")
    print(f"Tickers : {', '.join(t[0] for t in TICKERS)}")
    print(f"Ticks   : {TICKS}   |   Decision every: {DECISION_EVERY} ticks")
    print(f"Events  : {len(EVENTS)} scheduled\n")

    # Setup
    market    = MarketStore()
    strategy  = Strategy()
    reactor   = EventReactor(strategy)
    risk      = RiskGate()
    portfolio = PortfolioView(cash=SEED_CASH)

    # Recorder for dashboard / replay
    Path("runs").mkdir(exist_ok=True)
    ts = int(time.time())
    rec_path = Path(f"runs/sim_{ts}.jsonl")
    recorder = RunRecorder(rec_path)

    # Generate all price data up front
    price_data = generate_prices(TICKS)

    # Build event lookup
    event_at: dict[int, dict] = {tick: payload for tick, payload in EVENTS}

    tick = 0
    all_decisions: list[dict] = []

    for t in range(TICKS):
        tick = t + 1

        # --- feed one tick of PRICE_UPDATE for every ticker ---
        for ticker, _, _, _, _ in TICKERS:
            price, volume = price_data[ticker][t]
            payload = {"ticker": ticker, "price": price, "volume": volume}
            market.apply_price_update(payload)
            recorder.record_ws("PRICE_UPDATE", payload)

        # --- inject scheduled event ---
        if t in event_at:
            evt = event_at[t]
            recorder.record_ws("MARKET_EVENT", evt)
            print(f"\n  {_colour('EVENT', YELLOW)}  tick={tick:>4}  "
                  f"{_colour(evt['event_type'], BOLD)}  \"{evt['headline']}\"")

            event_intents = reactor.react(evt, market, portfolio)
            if event_intents:
                allowed = risk.filter(event_intents, portfolio, market)
                for intent in allowed:
                    state = market.get(intent.ticker)
                    if not state:
                        continue
                    fill_price = Decimal(str(state.price))
                    apply_fill(portfolio, intent, fill_price)
                    tag = _colour("BUY ", GREEN) if intent.side == "BUY" else _colour("SELL", RED)
                    print(f"  {tag}  {intent.ticker:<6}  qty={intent.quantity:<5} "
                          f"@{fill_price:.2f}  [{intent.reason}]")
                    recorder.record_order(
                        side=intent.side, ticker=intent.ticker, qty=intent.quantity,
                        order_type=intent.order_type, limit_price=intent.limit_price,
                        reason=intent.reason, result={"simulated": True},
                    )
                    all_decisions.append({
                        "tick": tick, "side": intent.side, "ticker": intent.ticker,
                        "qty": intent.quantity, "price": float(fill_price), "reason": intent.reason,
                    })

        # --- strategy decision cycle every DECISION_EVERY ticks ---
        if tick % DECISION_EVERY == 0:
            intents = strategy.decide(market, portfolio)
            allowed = risk.filter(intents, portfolio, market)
            for intent in allowed:
                state = market.get(intent.ticker)
                if not state:
                    continue
                fill_price = Decimal(str(state.price))
                apply_fill(portfolio, intent, fill_price)
                tag = _colour("BUY ", GREEN) if intent.side == "BUY" else _colour("SELL", RED)
                eq = portfolio.equity(market)
                print(f"  {tag}  {intent.ticker:<6}  qty={intent.quantity:<5} "
                      f"@{fill_price:.2f}  [{intent.reason}]  "
                      f"equity=${eq:,.0f}  tick={tick}")
                recorder.record_order(
                    side=intent.side, ticker=intent.ticker, qty=intent.quantity,
                    order_type=intent.order_type, limit_price=intent.limit_price,
                    reason=intent.reason, result={"simulated": True},
                )
                all_decisions.append({
                    "tick": tick, "side": intent.side, "ticker": intent.ticker,
                    "qty": intent.quantity, "price": float(fill_price), "reason": intent.reason,
                })

    recorder.close()

    # ---------------------------------------------------------------- summary
    final_equity = portfolio.equity(market)
    gain = final_equity - SEED_CASH
    gain_pct = gain / SEED_CASH * 100

    colour = GREEN if gain >= 0 else RED

    print(f"\n{BOLD}{'='*55}{RESET}")
    print(f"{BOLD}SIMULATION COMPLETE{RESET}")
    print(f"{'='*55}")
    print(f"  Ticks processed : {TICKS:,}")
    print(f"  Decisions taken : {len(all_decisions)}")
    print(f"  Open positions  : {len(portfolio.positions)}")
    print(f"  Cash remaining  : ${portfolio.cash:,.2f}")
    print(f"  Final equity    : ${final_equity:,.2f}")
    print(f"  P&L             : {_colour(f'${gain:+,.2f} ({gain_pct:+.2f}%)', colour)}")

    if portfolio.positions:
        print(f"\n  Open positions:")
        for ticker, pos in portfolio.positions.items():
            state = market.get(ticker)
            last = Decimal(str(state.price)) if state else pos.avg_cost
            pnl = (last - pos.avg_cost) / pos.avg_cost * 100
            c = GREEN if pnl >= 0 else RED
            print(f"    {ticker:<6}  qty={pos.quantity:<5}  "
                  f"avg=${pos.avg_cost:.2f}  last=${last:.2f}  "
                  f"P&L={_colour(f'{pnl:+.1f}%', c)}")

    if all_decisions:
        print(f"\n  Last 5 decisions:")
        for d in all_decisions[-5:]:
            tag = _colour("BUY ", GREEN) if d["side"] == "BUY" else _colour("SELL", RED)
            print(f"    tick={d['tick']:>5}  {tag} {d['ticker']:<6} "
                  f"qty={d['qty']:<5} @{d['price']:.2f}  [{d['reason']}]")

    print(f"\n  Replay file: {rec_path}")
    print(f"  Run replay : python -m bot replay {rec_path}")
    print(f"  Dashboard  : streamlit run dashboards/app.py")
    print()


if __name__ == "__main__":
    run()
