"""Tiny tests that don't need the platform running."""
from __future__ import annotations

import os

import pytest


def test_market_store_handles_snake_and_camel_case():
    from bot.market import MarketStore

    store = MarketStore()
    store.seed_from_snapshot([
        {"ticker": "ARKA", "current_price": 100.0, "volume": 1000},
        {"ticker": "MNVS", "currentPrice": 50.5, "volume": 500},
    ])
    assert len(store) == 2
    assert store.get("ARKA").price == 100.0
    assert store.get("MNVS").price == 50.5

    # PRICE_UPDATE payload
    store.apply_price_update({
        "ticker": "ARKA",
        "price": 101.25,
        "change": 1.25,
        "change_pct": 1.25,
        "volume": 1500,
        "market_time": "2026-05-12T10:00:00",
    })
    arka = store.get("ARKA")
    assert arka.price == 101.25
    assert arka.change_pct == 1.25
    assert list(arka.recent)[-2:] == [100.0, 101.25]


def test_market_store_ignores_unknown_payload():
    from bot.market import MarketStore

    store = MarketStore()
    result = store.apply_price_update({"price": 100.0})  # no ticker
    assert result is None
    assert len(store) == 0


def test_settings_requires_credentials(monkeypatch):
    from bot.config import Settings

    monkeypatch.delenv("BOT_EMAIL", raising=False)
    monkeypatch.delenv("BOT_PASSWORD", raising=False)
    with pytest.raises(RuntimeError):
        Settings.from_env()


def test_settings_reads_env(monkeypatch):
    from bot.config import Settings

    monkeypatch.setenv("BOT_EMAIL", "bot@example.com")
    monkeypatch.setenv("BOT_PASSWORD", "pw")
    monkeypatch.setenv("SEED_DEPOSIT", "50000")
    monkeypatch.setenv("SEED_CURRENCY", "eur")
    s = Settings.from_env()
    assert s.bot_email == "bot@example.com"
    assert str(s.seed_deposit) == "50000"
    assert s.seed_currency == "eur"
    assert s.gateway_http_url.startswith("http")
