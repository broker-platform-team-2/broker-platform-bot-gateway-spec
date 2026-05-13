"""Tests for Executor reconcile-then-retry logic."""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from bot.executor import Executor, OpenOrder
from bot.strategy import OrderIntent

pytestmark = pytest.mark.anyio


def _make_intent(side: str = "BUY", ticker: str = "ARKA", qty: int = 10) -> OrderIntent:
    return OrderIntent(
        side=side,
        ticker=ticker,
        quantity=qty,
        order_type="LIMIT",
        limit_price=Decimal("100.10"),
        reason="entry",
    )


def _make_http(*, place_order_response=None, place_order_side_effect=None, ticker_qty: int = 0):
    http = MagicMock()
    if place_order_side_effect is not None:
        http.place_order = AsyncMock(side_effect=place_order_side_effect)
    else:
        http.place_order = AsyncMock(return_value=place_order_response or {
            "exchangeOrderId": "ORD-1", "status": "OPEN"
        })
    http.get_portfolio_ticker_qty = AsyncMock(return_value=ticker_qty)
    return http


async def test_submit_success():
    http = _make_http(place_order_response={"exchangeOrderId": "ORD-1", "status": "OPEN"})
    executor = Executor(http)
    oo = await executor.submit(_make_intent())
    assert oo is not None
    assert oo.order_id == "ORD-1"
    assert "ORD-1" in executor.open_orders


async def test_submit_4xx_returns_none():
    resp = MagicMock()
    resp.status_code = 400
    http = _make_http(
        place_order_side_effect=httpx.HTTPStatusError("bad", request=MagicMock(), response=resp)
    )
    executor = Executor(http)
    oo = await executor.submit(_make_intent())
    assert oo is None
    assert executor.open_orders == {}


async def test_timeout_triggers_reconcile_then_retry_when_not_placed():
    """On timeout: reconcile shows no position -> retry fires and succeeds."""
    attempt = {"n": 0}

    async def flaky_place(**kwargs):
        attempt["n"] += 1
        if attempt["n"] == 1:
            raise httpx.TimeoutException("timeout")
        return {"exchangeOrderId": "ORD-RETRY", "status": "OPEN"}

    http = MagicMock()
    http.place_order = AsyncMock(side_effect=flaky_place)
    # qty=0 -> order did NOT land -> retry
    http.get_portfolio_ticker_qty = AsyncMock(return_value=0)

    executor = Executor(http)
    with patch("bot.executor.asyncio.sleep", new=AsyncMock()):
        oo = await executor.submit(_make_intent())

    assert oo is not None
    assert oo.order_id == "ORD-RETRY"
    assert attempt["n"] == 2
    http.get_portfolio_ticker_qty.assert_awaited_once_with("ARKA")


async def test_timeout_no_retry_when_order_already_placed():
    """On timeout: reconcile shows position exists -> no retry."""
    attempt = {"n": 0}

    async def always_timeout(**kwargs):
        attempt["n"] += 1
        raise httpx.TimeoutException("timeout")

    http = MagicMock()
    http.place_order = AsyncMock(side_effect=always_timeout)
    # qty=10 -> order landed despite timeout
    http.get_portfolio_ticker_qty = AsyncMock(return_value=10)

    executor = Executor(http)
    with patch("bot.executor.asyncio.sleep", new=AsyncMock()):
        oo = await executor.submit(_make_intent(side="BUY"))

    assert oo is None  # reconcile confirmed placed; no retry needed
    assert attempt["n"] == 1  # only the initial attempt
    http.get_portfolio_ticker_qty.assert_awaited_once_with("ARKA")


async def test_5xx_triggers_reconcile_then_retry():
    """5xx response also triggers the reconcile-then-retry path."""
    attempt = {"n": 0}

    async def flaky_place(**kwargs):
        attempt["n"] += 1
        if attempt["n"] == 1:
            resp = MagicMock()
            resp.status_code = 503
            raise httpx.HTTPStatusError("service unavailable", request=MagicMock(), response=resp)
        return {"exchangeOrderId": "ORD-AFTER-503", "status": "OPEN"}

    http = MagicMock()
    http.place_order = AsyncMock(side_effect=flaky_place)
    http.get_portfolio_ticker_qty = AsyncMock(return_value=0)  # not placed yet

    executor = Executor(http)
    with patch("bot.executor.asyncio.sleep", new=AsyncMock()):
        oo = await executor.submit(_make_intent())

    assert oo is not None
    assert oo.order_id == "ORD-AFTER-503"
    assert attempt["n"] == 2


async def test_sell_reconcile_confirms_placed_when_position_gone():
    """SELL timeout: portfolio shows qty=0 -> sell landed -> no retry."""
    async def always_timeout(**kwargs):
        raise httpx.TimeoutException("timeout")

    http = MagicMock()
    http.place_order = AsyncMock(side_effect=always_timeout)
    # qty=0 for a SELL means the position is gone -> order placed
    http.get_portfolio_ticker_qty = AsyncMock(return_value=0)

    executor = Executor(http)
    with patch("bot.executor.asyncio.sleep", new=AsyncMock()):
        oo = await executor.submit(_make_intent(side="SELL"))

    assert oo is None  # confirmed placed


def test_handle_order_update_removes_filled():
    http = _make_http()
    executor = Executor(http)
    executor.open_orders["ORD-1"] = OpenOrder(
        order_id="ORD-1", ticker="ARKA", side="BUY",
        quantity=10, limit_price=Decimal("100"), reason="entry",
    )
    executor.handle_order_update({"exchangeOrderId": "ORD-1", "status": "FILLED"})
    assert "ORD-1" not in executor.open_orders
