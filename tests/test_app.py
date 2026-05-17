"""Tests for Bot orchestration: reconcile, optimistic updates, event handling."""
from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.app import Bot
from bot.events import EventReactor
from bot.executor import Executor
from bot.market import MarketStore
from bot.orderbook import OrderBook
from bot.risk import RiskGate
from bot.strategy import OrderIntent, PortfolioView, Position, Strategy

pytestmark = pytest.mark.anyio


# ─────────────────────────────────────────────── helpers

def _make_bot() -> Bot:
    """Return a Bot with all external dependencies mocked."""
    settings = MagicMock()
    settings.seed_currency = "USD"

    bot = Bot.__new__(Bot)
    bot.settings = settings
    bot.http = AsyncMock()
    bot.market = MarketStore()
    bot.orderbook = OrderBook()
    bot.strategy = Strategy()
    bot.executor = MagicMock()
    bot.executor.submit_all = AsyncMock(return_value=[])
    bot.events = EventReactor(bot.strategy)
    bot._subscriber_portfolios = {}
    bot._subscriber_risk_gates = {}
    bot._pending_sells = {}
    bot._peak_prices = {}
    bot._price_update_counter = 0
    bot._decision_lock = asyncio.Lock()
    bot._reconcile_task = None
    return bot


def _make_accounts(balance: float = 10000.0, currency: str = "USD") -> list[dict]:
    return [{"currency": currency, "balance": str(balance)}]


def _make_holdings(ticker: str, qty: int, avg: float) -> list[dict]:
    return [{"instrumentId": ticker, "amount": qty, "averageCost": str(avg)}]


# ─────────────────────────────────────────────── _reconcile_subscriber

async def test_reconcile_builds_portfolio_from_api():
    bot = _make_bot()
    bot.http.get_accounts = AsyncMock(return_value=_make_accounts(50000.0))
    bot.http.get_portfolio = AsyncMock(return_value=_make_holdings("AAPL", 10, 150.0))

    await bot._reconcile_subscriber(1)

    pf = bot._subscriber_portfolios[1]
    assert pf.cash == Decimal("50000")
    assert "AAPL" in pf.positions
    assert pf.positions["AAPL"].quantity == 10
    assert pf.positions["AAPL"].avg_cost == Decimal("150")


async def test_reconcile_preserves_peak_price_from_peak_tracker():
    bot = _make_bot()
    bot._peak_prices[1] = {"AAPL": Decimal("200")}
    bot.http.get_accounts = AsyncMock(return_value=_make_accounts(50000.0))
    bot.http.get_portfolio = AsyncMock(return_value=_make_holdings("AAPL", 10, 150.0))

    await bot._reconcile_subscriber(1)

    assert bot._subscriber_portfolios[1].positions["AAPL"].peak_price == Decimal("200")


async def test_reconcile_falls_back_to_avg_cost_when_no_peak():
    bot = _make_bot()
    bot.http.get_accounts = AsyncMock(return_value=_make_accounts(50000.0))
    bot.http.get_portfolio = AsyncMock(return_value=_make_holdings("AAPL", 10, 130.0))

    await bot._reconcile_subscriber(1)

    assert bot._subscriber_portfolios[1].positions["AAPL"].peak_price == Decimal("130")


async def test_reconcile_cleans_stale_peaks_for_closed_positions():
    bot = _make_bot()
    # AAPL was tracked, but it's now gone from the API portfolio
    bot._peak_prices[1] = {"AAPL": Decimal("200"), "TSLA": Decimal("300")}
    bot.http.get_accounts = AsyncMock(return_value=_make_accounts(50000.0))
    # Only TSLA still open
    bot.http.get_portfolio = AsyncMock(return_value=_make_holdings("TSLA", 5, 280.0))

    await bot._reconcile_subscriber(1)

    assert "AAPL" not in bot._peak_prices[1]
    assert "TSLA" in bot._peak_prices[1]


async def test_reconcile_skips_on_http_error():
    bot = _make_bot()
    bot.http.get_accounts = AsyncMock(side_effect=RuntimeError("network error"))

    await bot._reconcile_subscriber(1)

    assert 1 not in bot._subscriber_portfolios


# ─────────────────────────────────────────────── _apply_optimistic_update

def test_optimistic_sell_reduces_quantity():
    bot = _make_bot()
    bot._subscriber_portfolios[1] = PortfolioView(
        cash=Decimal("5000"),
        positions={"AAPL": Position("AAPL", 10, Decimal("100"), Decimal("110"))},
    )
    bot._peak_prices[1] = {"AAPL": Decimal("110")}
    sell = OrderIntent(side="SELL", ticker="AAPL", quantity=4, order_type="MARKET", reason="hard_stop")

    bot._apply_optimistic_update(1, [sell])

    pf = bot._subscriber_portfolios[1]
    assert pf.positions["AAPL"].quantity == 6
    assert bot._pending_sells[1]["AAPL"] == 4


def test_optimistic_sell_removes_position_when_fully_sold():
    bot = _make_bot()
    bot._subscriber_portfolios[1] = PortfolioView(
        cash=Decimal("5000"),
        positions={"AAPL": Position("AAPL", 10, Decimal("100"), Decimal("110"))},
    )
    bot._peak_prices[1] = {"AAPL": Decimal("110")}
    sell = OrderIntent(side="SELL", ticker="AAPL", quantity=10, order_type="MARKET", reason="trailing_stop")

    bot._apply_optimistic_update(1, [sell])

    pf = bot._subscriber_portfolios[1]
    assert "AAPL" not in pf.positions
    assert "AAPL" not in bot._peak_prices[1]


def test_optimistic_buy_adds_new_position():
    bot = _make_bot()
    bot.market.apply_price_update({"ticker": "NVDA", "price": 500.0, "volume": 1000})
    bot._subscriber_portfolios[1] = PortfolioView(cash=Decimal("20000"), positions={})

    buy = OrderIntent(side="BUY", ticker="NVDA", quantity=5, order_type="LIMIT",
                      limit_price=Decimal("500"), reason="entry")

    bot._apply_optimistic_update(1, [buy])

    pf = bot._subscriber_portfolios[1]
    assert "NVDA" in pf.positions
    assert pf.positions["NVDA"].quantity == 5
    assert pf.positions["NVDA"].peak_price == Decimal("500")
    assert bot._peak_prices[1]["NVDA"] == Decimal("500")


def test_optimistic_buy_increases_existing_position():
    bot = _make_bot()
    bot.market.apply_price_update({"ticker": "NVDA", "price": 520.0, "volume": 1000})
    bot._subscriber_portfolios[1] = PortfolioView(
        cash=Decimal("20000"),
        positions={"NVDA": Position("NVDA", 10, Decimal("500"), Decimal("510"))},
    )
    bot._peak_prices[1] = {"NVDA": Decimal("510")}
    buy = OrderIntent(side="BUY", ticker="NVDA", quantity=5, order_type="LIMIT",
                      limit_price=Decimal("520"), reason="pyramid")

    bot._apply_optimistic_update(1, [buy])

    pf = bot._subscriber_portfolios[1]
    assert pf.positions["NVDA"].quantity == 15
    assert pf.positions["NVDA"].peak_price == Decimal("520")
    assert bot._peak_prices[1]["NVDA"] == Decimal("520")


# ─────────────────────────────────────────────── _on_market_event

async def test_market_event_updates_strategy_scores():
    bot = _make_bot()
    for t, base in [("A", 100.0), ("B", 200.0), ("C", 50.0), ("D", 80.0), ("E", 60.0)]:
        for i in range(40):
            bot.market.apply_price_update({"ticker": t, "price": base + i * 0.5, "volume": 1000})

    bot.http.get_active_subscribers = AsyncMock(return_value=[])

    await bot._on_market_event({
        "event_type": "BULL_RUN", "scope": "MARKET", "target": None,
        "magnitude": 0.05, "headline": "test",
    })

    assert bot.strategy._last_scores != {}


async def test_market_event_calls_reactor_and_submits_orders():
    bot = _make_bot()
    for t, base in [("A", 100.0), ("B", 200.0), ("C", 50.0), ("D", 80.0), ("E", 60.0)]:
        for i in range(40):
            bot.market.apply_price_update({"ticker": t, "price": base + i * 0.5, "volume": 1000})

    bot._subscriber_portfolios[99] = PortfolioView(cash=Decimal("100000"), positions={})
    bot.http.get_active_subscribers = AsyncMock(return_value=[{"userId": 99}])
    bot._subscriber_risk_gates[99] = RiskGate()

    captured: list = []

    async def capture_submit(intents, on_behalf_of=None):
        captured.extend(intents)
        return []

    bot.executor.submit_all = capture_submit

    await bot._on_market_event({
        "event_type": "BULL_RUN", "scope": "MARKET", "target": None,
        "magnitude": 0.05, "headline": "test",
    })

    assert len(captured) > 0
    assert all(i.side == "BUY" for i in captured)


# ─────────────────────────────────────────────── peak_price survives reconcile

async def test_peak_price_survives_reconcile_cycle():
    """Peak prices written by strategy must survive the 30-s reconcile."""
    bot = _make_bot()

    # Seed market with enough data for scoring
    for t, base in [("A", 100.0), ("B", 200.0), ("C", 50.0), ("D", 80.0), ("E", 60.0)]:
        for i in range(40):
            bot.market.apply_price_update({"ticker": t, "price": base + i * 0.5, "volume": 1000})

    # Manually set a portfolio with a peak higher than avg_cost
    bot._subscriber_portfolios[1] = PortfolioView(
        cash=Decimal("100000"),
        positions={"A": Position("A", 10, Decimal("100"), Decimal("119"))},
    )
    bot._peak_prices[1] = {"A": Decimal("119")}

    # Reconcile arrives with the position still open, avg_cost unchanged
    bot.http.get_accounts = AsyncMock(return_value=_make_accounts(100000.0))
    bot.http.get_portfolio = AsyncMock(return_value=_make_holdings("A", 10, 100.0))

    await bot._reconcile_subscriber(1)

    # Peak price must NOT reset to avg_cost
    assert bot._subscriber_portfolios[1].positions["A"].peak_price == Decimal("119")
