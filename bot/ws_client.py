"""
Async WebSocket consumer for /notifications/ws.

One socket carries everything we need:
  * PRICE_UPDATE       — per-ticker tick (drives the market store)
  * ORDER_UPDATE       — fill / partial-fill / cancel for our own orders
  * MARKET_EVENT       — BULL_RUN / BEAR_CRASH / SECTOR_BOOM / SECTOR_SLUMP / STOCK_SHOCK
  * ORDER_BOOK_UPDATE  — top-of-book changes

Consumers register handlers per message type. The loop auto-reconnects with
exponential backoff so transient network blips don't kill the bot.
"""
from __future__ import annotations

import asyncio
import json
import random
import urllib.parse
from collections.abc import Awaitable, Callable
from typing import Any

import websockets

from .config import Settings
from .logging_setup import get_logger

log = get_logger(__name__)

Handler = Callable[[dict[str, Any]], Awaitable[None]]


class WsClient:
    def __init__(self, settings: Settings, get_token: Callable[[], str | None]) -> None:
        self._settings = settings
        self._get_token = get_token
        self._handlers: dict[str, list[Handler]] = {}
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    def on(self, message_type: str, handler: Handler) -> None:
        """Register an async handler for a message type (PRICE_UPDATE, ...)."""
        self._handlers.setdefault(message_type, []).append(handler)

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run_forever(), name="ws-consumer")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    # ----------------------------------------------------------------- internals
    async def _run_forever(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            try:
                await self._connect_and_consume()
                attempt = 0  # reset on a clean disconnect
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                delay = min(30.0, (2 ** attempt) + random.random())
                log.warning(
                    "ws.reconnect",
                    attempt=attempt,
                    delay_s=round(delay, 1),
                    error=str(exc),
                )
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass

    async def _connect_and_consume(self) -> None:
        token = self._get_token()
        if not token:
            raise RuntimeError("No JWT available — call HttpClient.authenticate() first")

        url = f"{self._settings.gateway_ws_url}?token={urllib.parse.quote(token)}"
        log.info("ws.connect", url=self._settings.gateway_ws_url)

        async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
            log.info("ws.connected")
            await self._consume(ws)

    async def _consume(self, ws: Any) -> None:
        async for raw in ws:
            if self._stop.is_set():
                return
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("ws.bad_json", raw=str(raw)[:200])
                continue

            msg_type = msg.get("type")
            payload = msg.get("payload") or {}
            handlers = self._handlers.get(msg_type, [])
            if not handlers:
                log.debug("ws.unhandled", type=msg_type)
                continue
            for handler in handlers:
                try:
                    await handler(payload)
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "ws.handler_error", type=msg_type, error=str(exc), exc_info=True
                    )
