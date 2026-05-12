"""Tests for the pure indicator functions — no platform needed."""
from __future__ import annotations

import numpy as np

from bot import indicators


def test_sma_returns_none_when_too_short():
    assert indicators.sma(np.array([1.0, 2.0]), length=5) is None


def test_sma_basic():
    closes = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    assert indicators.sma(closes, length=5) == 14.0
    assert indicators.sma(closes, length=3) == 16.0  # last 3: 14,16,18 -> 16


def test_ema_falls_close_to_sma_with_constant_input():
    closes = np.full(50, 100.0)
    result = indicators.ema(closes, length=10)
    assert result is not None
    assert abs(result - 100.0) < 1e-6


def test_atr_proxy_is_zero_for_flat_prices():
    closes = np.full(20, 100.0)
    result = indicators.atr_proxy(closes, length=14)
    assert result == 0.0


def test_atr_proxy_grows_with_volatility():
    quiet = np.array([100.0 + 0.01 * i for i in range(20)])
    wild = np.array([100.0 + 5.0 * (-1) ** i for i in range(20)])
    assert indicators.atr_proxy(wild, length=14) > indicators.atr_proxy(quiet, length=14)


def test_rate_of_change_positive_when_uptrend():
    closes = np.array([100.0 + i for i in range(15)])
    roc = indicators.rate_of_change(closes, length=10)
    assert roc is not None
    assert roc > 0


def test_breakout_strength_positive_when_new_high():
    # First 20 closes between 90-100, then a spike to 105
    closes = np.concatenate([
        np.linspace(95.0, 100.0, 20),
        np.array([105.0]),
    ])
    bs = indicators.breakout_strength(closes, length=20, atr_len=14)
    assert bs is not None
    assert bs > 0  # 105 > donchian high of 100


def test_momentum_score_returns_none_if_not_enough_data():
    closes = np.array([100.0, 101.0, 102.0])
    volumes = np.array([1000.0, 1100.0, 1200.0])
    assert indicators.momentum_score(closes, volumes) is None


def test_momentum_score_is_positive_for_uptrend():
    closes = np.array([100.0 + 0.5 * i for i in range(40)])
    volumes = np.full(40, 1000.0)
    score = indicators.momentum_score(closes, volumes)
    assert score is not None
    assert score > 0
