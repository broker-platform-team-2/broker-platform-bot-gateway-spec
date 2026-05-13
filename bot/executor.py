"""
Order executor.

Turns OrderIntent objects into POST /orders calls. Tracks open orders by
id so we can reconcile when an ORDER_UPDATE comes in over the WebSocket.

Reconcile-then-retry: if a POST times out or returns a 5xx we don't blindly
retry (there is no idempotency key on the platform). Instead we fetch
GET /portfolio to check whether the order actually landed, and only retry
if the position did NOT change in the expected direction.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import httpx

from .http_client import HttpClient
from .logging_setup import get_logger
from .strategy import OrderIntent

log = get_logger(__name__)


@dataclass
class OpenOrder:
    order_id: str
    ticker: str
    side: str
    quantity: int
    limit_price: Decimal | None
    reason: str
    raw: dict[str, Any] = field(default_factory=dict)


class Executor:
    def __init__(self, http: HttpClient) -> None:
        self.http = http
        self.open_orders: dict[str, OpenOrder] = {}

    async def submit(self, intent: OrderIntent) -> OpenOrder | None:
        """Place one order. Returns the OpenOrder, or None on failure.

        On timeout or 5xx: reconcile against /portfolio before deciding
        whether to retry, to avoid double-placing without an idempotency key.
        """
        try:
            resp = await self.http.place_order(
                instrument_type="STOCK",
                instrument_id=intent.ticker,
                order_type=intent.order_type,
                side=intent.side,
                quantity=intent.quantity,
                limit_price=intent.limit_price,
            )
        except httpx.TimeoutException:
            log.warning("exec.timeout", ticker=intent.ticker, side=intent.side, qty=intent.quantity)
            return await self._reconcile_then_retry(intent)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500:
                log.warning("exec.5xx", ticker=intent.ticker, status=exc.response.status_code)
                return await self._reconcile_then_retry(intent)
            log.error("exec.place_failed", ticker=intent.ticker, side=intent.side,
                      qty=intent.quantity, reason=intent.reason, error=str(exc))
            return None
        except Exception as exc:  # noqa: BLE001
            log.error("exec.place_failed", ticker=intent.ticker, side=intent.side,
                      qty=intent.quantity, reason=intent.reason, error=str(exc))
            return None

        return self._register_placed(intent, resp)

    async def submit_all(self, intents: list[OrderIntent]) -> list[OpenOrder]:
        results: list[OpenOrder] = []
        for intent in intents:
            placed = await self.submit(intent)
            if placed:
                results.append(placed)
        return results

    def handle_order_update(self, payload: dict[str, Any]) -> OpenOrder | None:
        """Called when an ORDER_UPDATE message arrives on the WebSocket."""
        order_id = str(
            payload.get("order_id")
            or payload.get("orderId")
            or payload.get("exchangeOrderId")
            or ""
        )
        if not order_id:
            return None
        oo = self.open_orders.get(order_id)
        if oo is None:
            return None
        status = payload.get("status")
        if status in ("FILLED", "CANCELLED", "REJECTED", "EXPIRED"):
            self.open_orders.pop(order_id, None)
            log.info("exec.closed", order_id=order_id, status=status, ticker=oo.ticker)
        return oo

    # ----------------------------------------------------------------- internals

    def _register_placed(self, intent: OrderIntent, resp: dict[str, Any]) -> OpenOrder | None:
        order_id = str(
            resp.get("exchangeOrderId")
            or resp.get("transactionId")
            or resp.get("orderId")
            or ""
        )
        if not order_id:
            log.warning("exec.no_order_id", response=resp)
            return None

        oo = OpenOrder(
            order_id=order_id,
            ticker=intent.ticker,
            side=intent.side,
            quantity=intent.quantity,
            limit_price=intent.limit_price,
            reason=intent.reason,
            raw=resp,
        )
        self.open_orders[order_id] = oo
        log.info("exec.placed", order_id=order_id, ticker=intent.ticker,
                 side=intent.side, qty=intent.quantity,
                 limit=str(intent.limit_price) if intent.limit_price else None,
                 reason=intent.reason)
        return oo

    async def _reconcile_then_retry(self, intent: OrderIntent) -> OpenOrder | None:
        """After a timeout or 5xx, check /portfolio before retrying."""
        await asyncio.sleep(1.0)

        held_qty = await self.http.get_portfolio_ticker_qty(intent.ticker)
        order_appears_placed = (
            (intent.side == "BUY" and held_qty > 0)
            or (intent.side == "SELL" and held_qty == 0)
        )

        if order_appears_placed:
            log.info("exec.reconcile.confirmed_placed", ticker=intent.ticker,
                     side=intent.side, held_qty=held_qty)
            return None

        log.info("exec.reconcile.retrying", ticker=intent.ticker, side=intent.side)
        try:
            resp = await self.http.place_order(
                instrument_type="STOCK",
                instrument_id=intent.ticker,
                order_type=intent.order_type,
                side=intent.side,
                quantity=intent.quantity,
                limit_price=intent.limit_price,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("exec.retry.failed", ticker=intent.ticker, side=intent.side, error=str(exc))
            return None

        return self._register_placed(intent, resp)
