"""Tests for WsClient: message dispatch, handler errors, reconnect backoff."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.ws_client import WsClient

pytestmark = pytest.mark.anyio


# ─────────────────────────────────────────────── helpers

def _settings() -> MagicMock:
    s = MagicMock()
    s.gateway_ws_url = "ws://fake/ws"
    return s


def _ws_client(token: str = "tok") -> WsClient:
    return WsClient(_settings(), get_token=lambda: token)


class _FakeWs:
    """Async context manager + async iterable over a list of raw JSON strings."""

    def __init__(self, raw_messages: list[str]) -> None:
        self._messages = raw_messages

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def __aiter__(self):
        return self._async_gen()

    async def _async_gen(self):
        for msg in self._messages:
            yield msg


def _wire_ws(messages: list[dict]) -> _FakeWs:
    """Return a fake websocket async-iterable that yields the given messages."""
    return _FakeWs([json.dumps(m) for m in messages])


# ─────────────────────────────────────────────── handler dispatch

async def test_registered_handler_is_called():
    wsc = _ws_client()
    received: list[dict] = []

    async def handler(payload: dict) -> None:
        received.append(payload)

    wsc.on("PRICE_UPDATE", handler)

    msg = {"type": "PRICE_UPDATE", "payload": {"ticker": "A", "price": 100}}
    fake_ws = _wire_ws([msg])

    with patch("bot.ws_client.websockets.connect", return_value=fake_ws):
        await wsc._connect_and_consume()

    assert len(received) == 1
    assert received[0]["ticker"] == "A"


async def test_unknown_message_type_does_not_raise():
    wsc = _ws_client()

    msg = {"type": "UNKNOWN_TYPE", "payload": {}}
    fake_ws = _wire_ws([msg])

    with patch("bot.ws_client.websockets.connect", return_value=fake_ws):
        await wsc._connect_and_consume()  # must not raise


async def test_malformed_json_does_not_crash():
    wsc = _ws_client()

    fake_ws = _FakeWs(["not-json", "also-not-json"])

    with patch("bot.ws_client.websockets.connect", return_value=fake_ws):
        await wsc._connect_and_consume()  # must not raise


async def test_handler_exception_does_not_crash_consume_loop():
    wsc = _ws_client()

    async def bad_handler(payload: dict) -> None:
        raise ValueError("intentional failure")

    wsc.on("PRICE_UPDATE", bad_handler)

    # Two messages — second one should still be dispatched
    good_received: list[dict] = []

    async def good_handler(payload: dict) -> None:
        good_received.append(payload)

    wsc.on("ORDER_UPDATE", good_handler)

    msgs = [
        {"type": "PRICE_UPDATE", "payload": {"ticker": "A"}},
        {"type": "ORDER_UPDATE", "payload": {"status": "FILLED"}},
    ]
    fake_ws = _wire_ws(msgs)

    with patch("bot.ws_client.websockets.connect", return_value=fake_ws):
        await wsc._connect_and_consume()

    assert len(good_received) == 1


async def test_multiple_handlers_for_same_type_all_called():
    wsc = _ws_client()
    results: list[str] = []

    async def h1(p: dict) -> None:
        results.append("h1")

    async def h2(p: dict) -> None:
        results.append("h2")

    wsc.on("PRICE_UPDATE", h1)
    wsc.on("PRICE_UPDATE", h2)

    fake_ws = _wire_ws([{"type": "PRICE_UPDATE", "payload": {}}])

    with patch("bot.ws_client.websockets.connect", return_value=fake_ws):
        await wsc._connect_and_consume()

    assert results == ["h1", "h2"]


# ─────────────────────────────────────────────── reconnect backoff

async def test_run_forever_reconnects_on_connection_error():
    wsc = _ws_client()
    attempt_counts = {"n": 0}

    async def fake_connect_and_consume():
        attempt_counts["n"] += 1
        if attempt_counts["n"] < 3:
            raise ConnectionError("dropped")
        # On 3rd attempt, signal stop so the loop exits
        wsc._stop.set()

    with patch.object(wsc, "_connect_and_consume", new=AsyncMock(side_effect=fake_connect_and_consume)):
        with patch("bot.ws_client.asyncio.wait_for", new=AsyncMock(side_effect=asyncio.TimeoutError)):
            await wsc._run_forever()

    assert attempt_counts["n"] == 3


async def test_start_and_stop_lifecycle():
    wsc = _ws_client()

    # Patch _run_forever so it returns immediately
    wsc._run_forever = AsyncMock()

    await wsc.start()
    assert wsc._task is not None

    await wsc.stop()
    assert wsc._task is None


async def test_start_is_idempotent():
    wsc = _ws_client()
    wsc._run_forever = AsyncMock()

    await wsc.start()
    first_task = wsc._task
    await wsc.start()  # second call should no-op

    assert wsc._task is first_task
    await wsc.stop()


# ─────────────────────────────────────────────── recorder

async def test_recorder_receives_messages():
    wsc = _ws_client()
    recorder = MagicMock()
    recorder.record_ws = MagicMock()
    wsc.set_recorder(recorder)

    msg = {"type": "PRICE_UPDATE", "payload": {"ticker": "X", "price": 10}}
    fake_ws = _wire_ws([msg])

    with patch("bot.ws_client.websockets.connect", return_value=fake_ws):
        await wsc._connect_and_consume()

    recorder.record_ws.assert_called_once_with("PRICE_UPDATE", {"ticker": "X", "price": 10})


async def test_recorder_error_does_not_crash():
    wsc = _ws_client()
    recorder = MagicMock()
    recorder.record_ws = MagicMock(side_effect=IOError("disk full"))
    wsc.set_recorder(recorder)

    fake_ws = _wire_ws([{"type": "PRICE_UPDATE", "payload": {}}])

    with patch("bot.ws_client.websockets.connect", return_value=fake_ws):
        await wsc._connect_and_consume()  # must not raise
