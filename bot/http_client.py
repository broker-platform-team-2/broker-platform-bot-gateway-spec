"""
Async HTTP client for the broker platform user-gateway.

Handles:
  * register-on-first-run (POST /users/register) -> JWT
  * login (POST /users/login)                    -> JWT
  * automatic Authorization header on every request
  * one re-login retry on 401
  * helpers for the endpoints the bot uses: accounts, funds, portfolio,
    market snapshot, orders.
"""
from __future__ import annotations

import asyncio
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
        self._auth_lock = asyncio.Lock()
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
        try:
            self._token = await self._login()
            log.info("auth.login.ok", email=self._settings.bot_email)
            return self._token
        except httpx.HTTPStatusError as exc:
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
            raise AuthError(f"Register failed with HTTP {r.status_code}: {r.text}")
        return r.json()["token"]

    # -------------------------------------------------------------- core helpers
    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        params: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        headers = {**self._auth_headers(), **(extra_headers or {})}
        r = await self._client.request(method, path, json=json, params=params, headers=headers)
        if r.status_code == 401:
            log.warning("http.401.relogin", path=path)
            self._token = None  # signal that any concurrent waiter should re-auth too
            await self._reauth()
            if self._token:  # only retry if re-auth succeeded
                headers = {**self._auth_headers(), **(extra_headers or {})}
                r = await self._client.request(method, path, json=json, params=params, headers=headers)
        return r

    async def _reauth(self) -> None:
        """Re-authenticate, serialising concurrent callers under a lock.

        If another coroutine already refreshed the token while this one waited
        for the lock, the refresh is skipped (the new token will be used on
        retry).  If authenticate() itself fails the error is logged but NOT
        re-raised so callers receive the original 401 response and their own
        error-handling (broad except / raise_for_status) decides what to do —
        the bot will retry on the next tick rather than freezing permanently.
        """
        async with self._auth_lock:
            if self._token:
                return  # refreshed by a concurrent coroutine while we waited
            try:
                await self.authenticate()
            except Exception as exc:  # noqa: BLE001
                log.error("http.reauth.failed", error=str(exc))

    def _auth_headers(self) -> dict[str, str]:
        if not self._token:
            return {}
        return {"Authorization": f"Bearer {self._token}"}

    def _behalf_headers(self, on_behalf_of: int | None) -> dict[str, str]:
        if on_behalf_of is None:
            return {}
        return {"X-On-Behalf-Of": str(on_behalf_of)}

    # ---------------------------------------------------------- endpoint helpers
    async def get_accounts(self, on_behalf_of: int | None = None) -> list[dict[str, Any]]:
        r = await self._request("GET", "/accounts/me",
                                extra_headers=self._behalf_headers(on_behalf_of))
        r.raise_for_status()
        return r.json()

    async def deposit(self, currency: str, amount: Decimal) -> dict[str, Any]:
        r = await self._request("POST", "/funds/deposit",
                                json={"currency": currency, "amount": str(amount)})
        r.raise_for_status()
        return r.json()

    async def get_portfolio(self, on_behalf_of: int | None = None) -> list[dict[str, Any]]:
        r = await self._request("GET", "/portfolio",
                                extra_headers=self._behalf_headers(on_behalf_of))
        r.raise_for_status()
        return r.json()

    async def get_portfolio_ticker_qty(self, ticker: str, on_behalf_of: int | None = None) -> int:
        try:
            holdings = await self.get_portfolio(on_behalf_of=on_behalf_of)
        except Exception:  # noqa: BLE001
            return 0
        for h in holdings:
            t = h.get("instrumentId") or h.get("ticker")
            if t == ticker:
                try:
                    return int(h.get("amount") or h.get("quantity") or 0)
                except (TypeError, ValueError):
                    return 0
        return 0

    async def get_active_subscribers(self) -> list[dict[str, Any]]:
        try:
            r = await self._request("GET", "/bots/active-subscribers")
            r.raise_for_status()
            return r.json()
        except Exception:  # noqa: BLE001
            return []

    async def get_bot_enabled(self) -> bool:
        try:
            r = await self._request("GET", "/bots/any-active")
            r.raise_for_status()
            return bool(r.json().get("enabled", False))
        except Exception:  # noqa: BLE001
            return False

    async def get_market_snapshot(self) -> list[dict[str, Any]]:
        r = await self._request("GET", "/exchange/market/stocks")
        r.raise_for_status()
        data = r.json()
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
        on_behalf_of: int | None = None,
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
        r = await self._request("POST", "/orders", json=body,
                                extra_headers=self._behalf_headers(on_behalf_of))
        r.raise_for_status()
        return r.json()

    async def get_order(self, order_id: str | int) -> dict[str, Any]:
        r = await self._request("GET", f"/orders/{order_id}")
        r.raise_for_status()
        return r.json()

    async def cancel_order(self, order_id: str | int) -> None:
        r = await self._request("DELETE", f"/orders/{order_id}")
        r.raise_for_status()
