"""Tests for HttpClient: auth flow, 401 re-login, and endpoint helpers."""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from bot.http_client import AuthError, HttpClient

pytestmark = pytest.mark.anyio


# ─────────────────────────────────────────────── helpers

def _settings(url: str = "http://fake") -> MagicMock:
    s = MagicMock()
    s.gateway_http_url = url
    s.bot_email = "bot@test.com"
    s.bot_password = "secret"
    s.bot_username = "testbot"
    return s


def _ok_response(body: object, status: int = 200) -> MagicMock:
    r = MagicMock(spec=httpx.Response)
    r.status_code = status
    r.json.return_value = body
    r.raise_for_status = MagicMock()
    r.text = str(body)
    return r


def _error_response(status: int, text: str = "error") -> MagicMock:
    r = MagicMock(spec=httpx.Response)
    r.status_code = status
    r.text = text
    r.raise_for_status.side_effect = httpx.HTTPStatusError(
        text, request=MagicMock(), response=r
    )
    return r


# ─────────────────────────────────────────────── authenticate

async def test_authenticate_login_success():
    client = HttpClient(_settings())
    login_resp = _ok_response({"token": "jwt-abc"})

    with patch.object(client._client, "request", new=AsyncMock(return_value=login_resp)):
        token = await client.authenticate()

    assert token == "jwt-abc"
    assert client.token == "jwt-abc"


async def test_authenticate_login_404_falls_back_to_register():
    client = HttpClient(_settings())
    fail_resp = _error_response(404)
    reg_resp = _ok_response({"token": "jwt-reg"})

    responses = [fail_resp, reg_resp]
    with patch.object(client._client, "request", new=AsyncMock(side_effect=responses)):
        token = await client.authenticate()

    assert token == "jwt-reg"
    assert client.token == "jwt-reg"


async def test_authenticate_5xx_raises_auth_error():
    client = HttpClient(_settings())
    resp = _error_response(503)

    with patch.object(client._client, "request", new=AsyncMock(return_value=resp)):
        with pytest.raises(AuthError):
            await client.authenticate()


async def test_authenticate_network_error_raises_auth_error():
    client = HttpClient(_settings())

    with patch.object(
        client._client, "request",
        new=AsyncMock(side_effect=httpx.ConnectError("refused")),
    ):
        with pytest.raises(AuthError):
            await client.authenticate()


# ─────────────────────────────────────────────── 401 re-login

async def test_request_retries_after_401():
    client = HttpClient(_settings())
    client._token = "old-token"

    unauth = _ok_response({}, status=401)
    unauth.status_code = 401
    unauth.raise_for_status = MagicMock()

    ok = _ok_response([{"currency": "USD", "balance": "5000"}])

    call_count = {"n": 0}

    async def fake_request(method, path, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return unauth
        if path == "/users/login":
            return _ok_response({"token": "new-token"})
        return ok

    with patch.object(client._client, "request", new=AsyncMock(side_effect=fake_request)):
        resp = await client._request("GET", "/accounts/me")

    assert client.token == "new-token"
    assert resp.status_code == 200


# ─────────────────────────────────────────────── endpoint helpers

async def test_get_portfolio_ticker_qty_returns_correct_qty():
    client = HttpClient(_settings())
    client._token = "tok"

    holdings = [
        {"instrumentId": "AAPL", "amount": 15},
        {"instrumentId": "TSLA", "amount": 5},
    ]
    with patch.object(client, "get_portfolio", new=AsyncMock(return_value=holdings)):
        qty = await client.get_portfolio_ticker_qty("AAPL")

    assert qty == 15


async def test_get_portfolio_ticker_qty_returns_zero_for_missing():
    client = HttpClient(_settings())
    client._token = "tok"

    with patch.object(client, "get_portfolio", new=AsyncMock(return_value=[])):
        qty = await client.get_portfolio_ticker_qty("NVDA")

    assert qty == 0


async def test_get_portfolio_ticker_qty_returns_zero_on_network_error():
    client = HttpClient(_settings())
    client._token = "tok"

    with patch.object(
        client, "get_portfolio",
        new=AsyncMock(side_effect=RuntimeError("network")),
    ):
        qty = await client.get_portfolio_ticker_qty("NVDA")

    assert qty == 0


async def test_get_active_subscribers_returns_empty_on_error():
    client = HttpClient(_settings())
    client._token = "tok"
    err_resp = _error_response(500)

    with patch.object(client._client, "request", new=AsyncMock(return_value=err_resp)):
        result = await client.get_active_subscribers()

    assert result == []


async def test_place_order_sends_correct_body():
    client = HttpClient(_settings())
    client._token = "tok"

    captured: list[dict] = []

    async def fake_request(method, path, **kwargs):
        captured.append({"method": method, "path": path, "json": kwargs.get("json")})
        return _ok_response({"exchangeOrderId": "X1", "status": "OPEN"})

    with patch.object(client._client, "request", new=AsyncMock(side_effect=fake_request)):
        resp = await client.place_order(
            instrument_type="STOCK",
            instrument_id="AAPL",
            order_type="LIMIT",
            side="BUY",
            quantity=10,
            limit_price=Decimal("150.50"),
        )

    assert resp["exchangeOrderId"] == "X1"
    body = captured[0]["json"]
    assert body["instrumentId"] == "AAPL"
    assert body["side"] == "BUY"
    assert body["quantity"] == 10
    assert body["limitPrice"] == "150.50"


async def test_get_market_snapshot_handles_list_response():
    client = HttpClient(_settings())
    client._token = "tok"

    stocks = [{"ticker": "A", "currentPrice": 100}]
    with patch.object(client._client, "request", new=AsyncMock(return_value=_ok_response(stocks))):
        result = await client.get_market_snapshot()

    assert result == stocks


async def test_get_market_snapshot_handles_wrapped_response():
    client = HttpClient(_settings())
    client._token = "tok"

    stocks = [{"ticker": "A", "currentPrice": 100}]
    with patch.object(
        client._client, "request",
        new=AsyncMock(return_value=_ok_response({"stocks": stocks})),
    ):
        result = await client.get_market_snapshot()

    assert result == stocks
