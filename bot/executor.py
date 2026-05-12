"""
Order executor.

Turns OrderIntent objects into POST /orders calls. Tracks open orders by
id so we can reconcile when an ORDER_UPDATE comes in over the WebSocket.

There is no idempotency key on the platform (yet) — if a POST times out
we DO NOT blindly retry. We let the ORDER_UPDATE / portfolio reconcile
catch it on the next tick.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

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
        """Place one order. Returns the OpenOrder, or None on failure."""
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
            log.error(
                "exec.place_failed",
                ticker=intent.ticker,
                side=intent.side,
                qty=intent.quantity,
                reason=intent.reason,
                error=str(exc),
            )
            return None

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
        log.info(
            "exec.placed",
            order_id=order_id,
            ticker=intent.ticker,
            side=intent.side,
            qty=intent.quantity,
            limit=str(intent.limit_price) if intent.limit_price else None,
            reason=intent.reason,
        )
        return oo

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
