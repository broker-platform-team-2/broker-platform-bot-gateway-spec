"""
Async HTTP client for the broker platform user-gateway.

Handles:
  * register-on-first-run (POST /users/register) → JWT
  * login (POST /users/login)                    → JWT
  * automatic Authorization header on every request
  * one re-login retry on 401
  * helpers for the endpoints the bot uses on Day 1 (accounts, funds,
    portfolio, market snapshot). Day 2 will add /orders.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

import httpx

from .config import Settings
from .logging_setup import get_logger

log = get_logger(__name__)


class AuthError(RuntimeError):
    """Raised when login + register both fail — bot cannot proceed."""


class HttpClient:
    def __init__(self, settings: Settings, *, timeout: float = 10.0) -> None:
        self._settings = settings
        self._token: str | None = None
        self._client = httpx.AsyncClient(
            base_url=settings.gateway_http_url,
            timeout=timeout,
        )

    # ------------------------------------------------------------------ lifecycle
    async def aclose(self) -> None:
        await self._client.aclose()

    @property
    def token(self) -> str | None:
        return self._token

    # ----------------------------------------------------------------------- auth
    async def authenticate(self) -> str:
        """Log in; if the account does not exist, register first. Returns JWT."""
        try:
            self._token = await self._login()
            log.info("auth.login.ok", email=self._settings.bot_email)
            return self._token
        except httpx.HTTPStatusError as exc:
            # 400/401/404 from login → assume the bot account doesn't exist yet
            if exc.response.status_code in (400, 401, 403, 404):
                log.info("auth.login.miss_register", status=exc.response.status_code)
                self._token = await self._register()
                log.info("auth.register.ok", email=self._settings.bot_email)
                return self._token
            raise AuthError(
                f"Login failed with HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            raise AuthError(
                f"Cannot reach broker platform at {self._settings.gateway_http_url}: {exc}"
            ) from exc

    async def _login(self) -> str:
        r = await self._client.post(
            "/users/login",
            json={
                "email": self._settings.bot_email,
                "password": self._settings.bot_password,
            },
        )
        r.raise_for_status()
        return r.json()["token"]

    async def _register(self) -> str:
        r = await self._client.post(
            "/users/register",
            json={
                "email": self._settings.bot_email,
                "username": self._settings.bot_username,
                "password": self._settings.bot_password,
            },
        )
        if r.status_code >= 400:
            raise AuthError(
                f"Register failed with HTTP {r.status_code}: {r.text}"
            )
        return r.json()["token"]

    # -------------------------------------------------------------- core helpers
    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        headers = self._auth_headers()
        r = await self._client.request(method, path, json=json, params=params, headers=headers)

        # Re-login once on 401 (token expired / rotated)
        if r.status_code == 401 and self._token is not None:
            log.warning("http.401.relogin", path=path)
            self._token = None
            await self.authenticate()
            r = await self._client.request(
                method, path, json=json, params=params, headers=self._auth_headers()
            )
        return r

    def _auth_headers(self) -> dict[str, str]:
        if not self._token:
            return {}
        return {"Authorization": f"Bearer {self._token}"}

    # ---------------------------------------------------------- endpoint helpers
    async def get_accounts(self) -> list[dict[str, Any]]:
        r = await self._request("GET", "/accounts/me")
        r.raise_for_status()
        return r.json()

    async def deposit(self, currency: str, amount: Decimal) -> dict[str, Any]:
        r = await self._request(
            "POST",
            "/funds/deposit",
            json={"currency": currency, "amount": str(amount)},
        )
        r.raise_for_status()
        return r.json()

    async def get_portfolio(self) -> list[dict[str, Any]]:
        r = await self._request("GET", "/portfolio")
        r.raise_for_status()
        return r.json()

    async def get_market_snapshot(self) -> list[dict[str, Any]]:
        """Initial /market/stocks snapshot via the gateway's /exchange proxy."""
        r = await self._request("GET", "/exchange/market/stocks")
        r.raise_for_status()
        data = r.json()
        # The exchange may return a bare list or {stocks: [...]} — handle both.
        if isinstance(data, list):
            return data
        for key in ("stocks", "data", "items"):
            if isinstance(data.get(key), list):
                return data[key]
        return []

    # -------------------------------------------------------------- orders
    async def place_order(
        self,
        *,
        instrument_type: str,
        instrument_id: str,
        order_type: str,
        side: str,
        quantity: int,
        limit_price: Decimal | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "instrumentType": instrument_type,
            "instrumentId": instrument_id,
            "orderType": order_type,
            "side": side,
            "quantity": quantity,
        }
        if limit_price is not None:
            body["limitPrice"] = str(limit_price)
        if expires_at is not None:
            body["expiresAt"] = expires_at
        r = await self._request("POST", "/orders", json=body)
        r.raise_for_status()
        return r.json()

    async def get_order(self, order_id: str | int) -> dict[str, Any]:
        r = await self._request("GET", f"/orders/{order_id}")
        r.raise_for_status()
        return r.json()

    async def cancel_order(self, order_id: str | int) -> None:
        r = await self._request("DELETE", f"/orders/{order_id}")
        r.raise_for_status()
