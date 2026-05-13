"""Tests for the OrderBook cache."""
from __future__ import annotations

from decimal import Decimal

from bot.orderbook import OrderBook


def test_apply_update_camelcase():
    ob = OrderBook()
    ob.apply_update({"ticker": "ARKA", "bestBid": "99.50", "bestAsk": "100.10"})
    assert ob.best_ask("ARKA") == Decimal("100.10")
    assert ob.best_bid("ARKA") == Decimal("99.50")


def test_apply_update_snakecase():
    ob = OrderBook()
    ob.apply_update({"ticker": "MNVS", "best_bid": 49.00, "best_ask": 49.50})
    assert ob.best_ask("MNVS") == Decimal("49.50")
    assert ob.best_bid("MNVS") == Decimal("49.00")


def test_missing_ticker_returns_none():
    ob = OrderBook()
    assert ob.best_ask("UNKN") is None
    assert ob.best_bid("UNKN") is None


def test_no_ticker_field_ignored():
    ob = OrderBook()
    ob.apply_update({"bestAsk": 100.0})  # no ticker key
    assert ob.tickers() == []


def test_partial_update_preserves_other_side():
    ob = OrderBook()
    ob.apply_update({"ticker": "ARKA", "bestBid": "99.00", "bestAsk": "100.00"})
    ob.apply_update({"ticker": "ARKA", "bestAsk": "100.50"})  # only ask updated
    assert ob.best_bid("ARKA") == Decimal("99.00")
    assert ob.best_ask("ARKA") == Decimal("100.50")


def test_tickers_lists_all_known():
    ob = OrderBook()
    ob.apply_update({"ticker": "AAA", "bestAsk": 10.0})
    ob.apply_update({"ticker": "BBB", "bestBid": 20.0})
    assert set(ob.tickers()) == {"AAA", "BBB"}


def test_invalid_price_ignored():
    ob = OrderBook()
    ob.apply_update({"ticker": "ARKA", "bestAsk": "not-a-number"})
    assert ob.best_ask("ARKA") is None
