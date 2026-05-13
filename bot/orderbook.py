"""Order book cache.

Stores the top-of-book (best bid / best ask) per ticker, fed by
ORDER_BOOK_UPDATE WebSocket messages. Strategy consults best_ask()
when placing entry LIMIT orders so we hit the real market price
instead of estimating price * 1.001.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any


class OrderBook:
    """In-memory top-of-book cache. Safe for single-event-loop use."""

    def __init__(self) -> None:
        self._best_bid: dict[str, Decimal] = {}
        self._best_ask: dict[str, Decimal] = {}

    def apply_update(self, payload: dict[str, Any]) -> None:
        ticker = payload.get("ticker") or payload.get("symbol")
        if not ticker:
            return
        bid = payload.get("bestBid") or payload.get("best_bid")
        ask = payload.get("bestAsk") or payload.get("best_ask")
        try:
            if bid is not None:
                self._best_bid[ticker] = Decimal(str(bid))
        except Exception:  # noqa: BLE001
            pass
        try:
            if ask is not None:
                self._best_ask[ticker] = Decimal(str(ask))
        except Exception:  # noqa: BLE001
            pass

    def best_ask(self, ticker: str) -> Decimal | None:
        return self._best_ask.get(ticker)

    def best_bid(self, ticker: str) -> Decimal | None:
        return self._best_bid.get(ticker)

    def tickers(self) -> list[str]:
        return list(set(self._best_ask) | set(self._best_bid))
