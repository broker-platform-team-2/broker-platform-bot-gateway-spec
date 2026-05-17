"""Order book cache.

Stores the top-of-book (best bid / best ask) per ticker, fed by
ORDER_BOOK_UPDATE WebSocket messages.

Real exchange payload format (Go WS server):
    {
        "ticker": "NVDA",
        "bids": [{"price": 479.10, "quantity": 200}, ...],
        "asks": [{"price": 479.50, "quantity": 150}, ...]
    }

The previous implementation looked for bestBid/bestAsk keys that don't
exist in the actual payload — this version correctly parses the arrays.

New additions:
  * imbalance()  — (bid_size - ask_size) / (bid_size + ask_size), -1..+1
  * spread_pct() — (ask - bid) / mid, used as a liquidity quality filter
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any


class OrderBook:
    """In-memory top-of-book cache. Safe for single-event-loop use."""

    def __init__(self) -> None:
        self._best_bid: dict[str, Decimal] = {}
        self._best_ask: dict[str, Decimal] = {}
        self._best_bid_size: dict[str, int] = {}
        self._best_ask_size: dict[str, int] = {}

    def apply_update(self, payload: dict[str, Any]) -> None:
        ticker = payload.get("ticker") or payload.get("symbol")
        if not ticker:
            return

        bids = payload.get("bids") or []
        asks = payload.get("asks") or []

        # Parse bids array — take the first (best) level
        if bids and isinstance(bids[0], dict):
            bid_price = bids[0].get("price")
            bid_size  = bids[0].get("quantity") or bids[0].get("size")
        else:
            # Fallback: legacy single-value keys (never sent by real exchange)
            bid_price = payload.get("bestBid") or payload.get("best_bid")
            bid_size  = payload.get("bestBidSize") or payload.get("best_bid_size")

        # Parse asks array — take the first (best) level
        if asks and isinstance(asks[0], dict):
            ask_price = asks[0].get("price")
            ask_size  = asks[0].get("quantity") or asks[0].get("size")
        else:
            ask_price = payload.get("bestAsk") or payload.get("best_ask")
            ask_size  = payload.get("bestAskSize") or payload.get("best_ask_size")

        try:
            if bid_price is not None:
                self._best_bid[ticker] = Decimal(str(bid_price))
        except Exception:  # noqa: BLE001
            pass
        try:
            if ask_price is not None:
                self._best_ask[ticker] = Decimal(str(ask_price))
        except Exception:  # noqa: BLE001
            pass
        try:
            if bid_size is not None:
                self._best_bid_size[ticker] = int(bid_size)
        except Exception:  # noqa: BLE001
            pass
        try:
            if ask_size is not None:
                self._best_ask_size[ticker] = int(ask_size)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ reads

    def best_ask(self, ticker: str) -> Decimal | None:
        return self._best_ask.get(ticker)

    def best_bid(self, ticker: str) -> Decimal | None:
        return self._best_bid.get(ticker)

    def imbalance(self, ticker: str) -> float | None:
        """Order book imbalance: (bid_size - ask_size) / (bid_size + ask_size).

        Range: -1.0 (all sellers) to +1.0 (all buyers).
        Positive means net buy pressure — consistent with simulation's
        order_pressure component driving prices up.
        Returns None when size data is not available.
        """
        bid_size = self._best_bid_size.get(ticker)
        ask_size = self._best_ask_size.get(ticker)
        if bid_size is None or ask_size is None:
            return None
        total = bid_size + ask_size
        if total == 0:
            return None
        return (bid_size - ask_size) / total

    def spread_pct(self, ticker: str) -> Decimal | None:
        """Bid-ask spread as a fraction of mid price.

        Used as a liquidity quality filter — wide spreads indicate
        poor fill quality or low activity in this simulation tick.
        Returns None when either side is missing.
        """
        bid = self._best_bid.get(ticker)
        ask = self._best_ask.get(ticker)
        if bid is None or ask is None or bid <= 0:
            return None
        mid = (bid + ask) / 2
        if mid <= 0:
            return None
        return (ask - bid) / mid

    def tickers(self) -> list[str]:
        return list(set(self._best_ask) | set(self._best_bid))
