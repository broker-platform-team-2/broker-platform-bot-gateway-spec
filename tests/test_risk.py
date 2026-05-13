"""Tests for the risk gate caps."""
from __future__ import annotations

from decimal import Decimal

from bot.market import MarketStore
from bot.risk import RiskGate
from bot.strategy import OrderIntent, PortfolioView


def _market_with(ticker: str, price: float) -> MarketStore:
    store = MarketStore()
    store.apply_price_update({"ticker": ticker, "price": price, "volume": 1000.0})
    return store


def test_blocks_orders_over_max_notional():
    gate = RiskGate()
    market = _market_with("AAA", 100.0)
    pf = PortfolioView(cash=Decimal("10000"))  # equity = 10000

    # 20% notional → over the 10% per-order cap
    huge = OrderIntent(side="BUY", ticker="AAA", quantity=20, order_type="LIMIT",
                       limit_price=Decimal("100"), reason="entry")
    out = gate.filter([huge], pf, market)
    assert out == []


def test_allows_sells_always_under_normal_conditions():
    gate = RiskGate()
    market = _market_with("AAA", 100.0)
    pf = PortfolioView(cash=Decimal("10000"))

    sell = OrderIntent(side="SELL", ticker="AAA", quantity=999,
                       order_type="MARKET", reason="hard_stop")
    out = gate.filter([sell], pf, market)
    assert out == [sell]


def test_kill_switch_pauses_buys_after_drawdown():
    gate = RiskGate()
    market = _market_with("AAA", 100.0)
    # Build a session peak
    high_pf = PortfolioView(cash=Decimal("100000"))
    gate.filter([], high_pf, market)  # sets peak

    # Now equity drops 20% — past the -15% kill-switch
    low_pf = PortfolioView(cash=Decimal("80000"))
    buy = OrderIntent(side="BUY", ticker="AAA", quantity=10, order_type="LIMIT",
                      limit_price=Decimal("100"), reason="entry")
    out = gate.filter([buy], low_pf, market)
    assert out == []
    assert gate.trading_paused()
