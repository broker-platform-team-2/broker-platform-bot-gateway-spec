"""
Pure indicator functions. No I/O, no side effects — easy to test.

Every function takes a 1-D numpy array of close prices (oldest -> newest)
and returns a single float (or None if there is not enough data).

Conventions:
  * Returning None means "not enough data yet — skip this ticker this tick."
  * Inputs are assumed already trimmed to the relevant window where it
    matters; callers pass the most recent N closes.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------- moving averages

def sma(closes: np.ndarray, length: int) -> float | None:
    """Simple moving average over the last `length` closes."""
    if closes.size < length:
        return None
    return float(closes[-length:].mean())


def ema(closes: np.ndarray, length: int) -> float | None:
    """Exponential moving average (standard k = 2 / (length + 1))."""
    if closes.size < length:
        return None
    k = 2.0 / (length + 1)
    # Seed EMA with the SMA of the first window, then iterate.
    seed = closes[:length].mean()
    e = seed
    for price in closes[length:]:
        e = price * k + e * (1.0 - k)
    return float(e)


# ------------------------------------------------------------------------ volatility

def atr_proxy(closes: np.ndarray, length: int = 14) -> float | None:
    """
    ATR (Average True Range) approximation using close-to-close moves.

    We don't have high/low per tick in our PRICE_UPDATE stream, so we
    approximate with mean absolute return over `length` ticks, scaled by
    the current price. Result is in *price units*, like real ATR.
    """
    if closes.size < length + 1:
        return None
    diffs = np.abs(np.diff(closes[-(length + 1):]))
    return float(diffs.mean())


# ------------------------------------------------------------------------- channels

def donchian_high(closes: np.ndarray, length: int = 20) -> float | None:
    """Highest close over the last `length` ticks (exclusive of the most recent)."""
    if closes.size < length + 1:
        return None
    return float(closes[-(length + 1):-1].max())


# --------------------------------------------------------------------- composite

def rate_of_change(closes: np.ndarray, length: int = 10) -> float | None:
    """(price_now - price_n_ticks_ago) / price_n_ticks_ago. Returns a fraction."""
    if closes.size < length + 1:
        return None
    past = float(closes[-(length + 1)])
    if past <= 0:
        return None
    return float((closes[-1] - past) / past)


def volume_surge(volumes: np.ndarray, length: int = 30) -> float | None:
    """current_volume / mean(prior `length` volumes). >1 = above-average flow."""
    if volumes.size < length + 1:
        return None
    base = float(volumes[-(length + 1):-1].mean())
    if base <= 0:
        return None
    return float(volumes[-1] / base)


def breakout_strength(
    closes: np.ndarray,
    length: int = 20,
    atr_len: int = 14,
) -> float | None:
    """
    How many ATRs is the current close above the prior `length`-tick high.

    Positive = breakout to the upside, negative = below the channel.
    """
    high = donchian_high(closes, length)
    atr = atr_proxy(closes, atr_len)
    if high is None or atr is None or atr <= 0:
        return None
    return float((closes[-1] - high) / atr)


def momentum_score(
    closes: np.ndarray,
    volumes: np.ndarray,
) -> float | None:
    """
    Composite momentum score: 0.5 * rate_of_change + 0.3 * (volume_surge - 1)
    + 0.2 * breakout_strength.

    The three components are intentionally on similar scales (roughly
    -0.1 to +0.1 each in calm markets, larger during events). Returns
    None if any component cannot be computed yet.
    """
    roc = rate_of_change(closes, length=10)
    vs = volume_surge(volumes, length=30)
    bs = breakout_strength(closes, length=20, atr_len=14)
    if roc is None or vs is None or bs is None:
        return None
    return 0.5 * roc + 0.3 * (vs - 1.0) + 0.2 * bs
