"""
In-memory market state.

Tracks per-ticker close prices and per-tick volumes in bounded ring
buffers, plus sector for event-routing. Fed by PRICE_UPDATE messages
from the WebSocket and seeded once at boot from /market/stocks.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TickerState:
    ticker: str
    sector: str = ""
    price: float = 0.0
    change: float = 0.0
    change_pct: float = 0.0
    volume: float = 0.0
    market_time: str = ""
    recent: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    volumes: deque[float] = field(default_factory=lambda: deque(maxlen=120))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class MarketStore:
    """Stores the latest tick per ticker. Fed by PRICE_UPDATE messages."""

    def __init__(self) -> None:
        self._tickers: dict[str, TickerState] = {}

    # ------------------------------------------------------------------- writes
    def seed_from_snapshot(self, stocks: list[dict[str, Any]]) -> None:
        """Initialise from /exchange/market/stocks. Tolerates camel/snake case."""
        for s in stocks:
            ticker = s.get("ticker") or s.get("symbol")
            if not ticker:
                continue
            price = _to_float(
                s.get("current_price") or s.get("currentPrice") or s.get("price")
            )
            volume = _to_float(s.get("volume"))
            state = TickerState(
                ticker=ticker,
                sector=s.get("sector") or s.get("industry") or "",
                price=price,
                volume=volume,
                market_time=s.get("market_time") or s.get("marketTime") or "",
            )
            state.recent.append(price)
            state.volumes.append(volume)
            self._tickers[ticker] = state

    def apply_price_update(self, payload: dict[str, Any]) -> TickerState | None:
        ticker = payload.get("ticker") or payload.get("symbol")
        if not ticker:
            return None
        price = _to_float(payload.get("price") or payload.get("current_price"))
        state = self._tickers.get(ticker) or TickerState(ticker=ticker)
        state.price = price
        state.change = _to_float(payload.get("change"))
        state.change_pct = _to_float(payload.get("change_pct") or payload.get("changePct"))
        state.volume = _to_float(payload.get("volume"), state.volume)
        state.market_time = payload.get("market_time") or payload.get("marketTime") or state.market_time
        # Sector usually only arrives in the initial snapshot; preserve it.
        new_sector = payload.get("sector") or payload.get("industry")
        if new_sector:
            state.sector = new_sector
        state.recent.append(price)
        state.volumes.append(state.volume)
        self._tickers[ticker] = state
        return state

    # -------------------------------------------------------------------- reads
    def get(self, ticker: str) -> TickerState | None:
        return self._tickers.get(ticker)

    def tickers(self) -> list[str]:
        return list(self._tickers.keys())

    def tickers_in_sector(self, sector: str) -> list[str]:
        if not sector:
            return []
        s = sector.lower()
        return [t for t, st in self._tickers.items() if st.sector.lower() == s]

    def __len__(self) -> int:
        return len(self._tickers)
