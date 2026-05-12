"""Tests for the strategy decision logic."""
from __future__ import annotations

from decimal import Decimal

from bot.market import MarketStore
from bot.strategy import PortfolioView, Position, Strategy


def _stuff_market_with_uptrend(store: MarketStore, ticker: str, base: float = 100.0) -> None:
    # 40 ticks of steady uptrend to satisfy MIN_PRICES_FOR_SCORING.
    payloads = [
        {"ticker": ticker, "price": base + 0.5 * i, "volume": 1000.0}
        for i in range(40)
    ]
    for p in payloads:
        store.apply_price_update(p)


def test_empty_market_yields_no_intents():
    store = MarketStore()
    pf = PortfolioView(cash=Decimal("10000"))
    intents = Strategy().decide(store, pf)
    assert intents == []


def test_no_entries_when_too_few_candidates():
    store = MarketStore()
    _stuff_market_with_uptrend(store, "AAA")
    pf = PortfolioView(cash=Decimal("10000"))
    intents = Strategy().decide(store, pf)
    # Only 1 candidate — under MIN_TICKERS_FOR_ENTRY=5
    assert intents == []


def test_buys_top_movers_when_enough_candidates():
    store = MarketStore()
    for t, base in [("AAA", 100.0), ("BBB", 200.0), ("CCC", 50.0), ("DDD", 80.0), ("EEE", 60.0)]:
        _stuff_market_with_uptrend(store, t, base)
    pf = PortfolioView(cash=Decimal("100000"))
    intents = Strategy().decide(store, pf)
    # Should emit one BUY per ticker (we don't own any yet)
    assert len(intents) == 5
    assert all(i.side == "BUY" for i in intents)
    assert all(i.order_type == "LIMIT" for i in intents)
    assert all(i.reason == "entry" for i in intents)


def test_hard_stop_triggers_when_price_drops_below_floor():
    store = MarketStore()
    # First, build an uptrend to position level
    _stuff_market_with_uptrend(store, "AAA", base=100.0)
    # Now plunge the price below the -3% hard stop
    store.apply_price_update({"ticker": "AAA", "price": 90.0, "volume": 1000.0})

    pf = PortfolioView(
        cash=Decimal("10000"),
        positions={
            "AAA": Position(
                ticker="AAA",
                quantity=10,
                avg_cost=Decimal("100"),
                peak_price=Decimal("100"),
            )
        },
    )
    intents = Strategy().decide(store, pf)
    sells = [i for i in intents if i.side == "SELL" and i.ticker == "AAA"]
    assert len(sells) == 1
    assert sells[0].order_type == "MARKET"
    assert sells[0].reason == "hard_stop"
