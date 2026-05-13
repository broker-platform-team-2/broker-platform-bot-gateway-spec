"""Tests for the EventReactor."""
from __future__ import annotations

from decimal import Decimal

from bot.events import EventReactor
from bot.market import MarketStore
from bot.strategy import PortfolioView, Position, Strategy


def _seed(store: MarketStore, ticker: str, price: float, sector: str) -> None:
    store.seed_from_snapshot([
        {"ticker": ticker, "current_price": price, "sector": sector, "volume": 1000},
    ])


def _make_reactor(scores: dict[str, float] | None = None) -> tuple[EventReactor, Strategy]:
    strat = Strategy()
    if scores:
        strat._last_scores = scores
    return EventReactor(strat), strat


# ------------------------------------------------------------------------ BEAR

def test_bear_crash_flattens_all_positions():
    store = MarketStore()
    _seed(store, "AAA", 100.0, "Tech")
    _seed(store, "BBB", 200.0, "Energy")

    pf = PortfolioView(
        cash=Decimal("0"),
        positions={
            "AAA": Position(ticker="AAA", quantity=50, avg_cost=Decimal("100"), peak_price=Decimal("100")),
            "BBB": Position(ticker="BBB", quantity=10, avg_cost=Decimal("200"), peak_price=Decimal("200")),
        },
    )
    reactor, _ = _make_reactor()

    intents = reactor.react(
        {"event_type": "BEAR_CRASH", "scope": "MARKET"},
        store, pf,
    )
    assert len(intents) == 2
    assert all(i.side == "SELL" for i in intents)
    assert all(i.order_type == "MARKET" for i in intents)
    assert {i.ticker for i in intents} == {"AAA", "BBB"}


def test_sector_slump_flattens_only_matching_sector():
    store = MarketStore()
    _seed(store, "AAA", 100.0, "Tech")
    _seed(store, "BBB", 200.0, "Energy")

    pf = PortfolioView(
        cash=Decimal("0"),
        positions={
            "AAA": Position(ticker="AAA", quantity=50, avg_cost=Decimal("100"), peak_price=Decimal("100")),
            "BBB": Position(ticker="BBB", quantity=10, avg_cost=Decimal("200"), peak_price=Decimal("200")),
        },
    )
    reactor, _ = _make_reactor()

    intents = reactor.react(
        {"event_type": "SECTOR_SLUMP", "scope": "SECTOR", "target": "Tech"},
        store, pf,
    )
    assert len(intents) == 1
    assert intents[0].ticker == "AAA"
    assert intents[0].side == "SELL"


# ------------------------------------------------------------------------ BULL

def test_bull_run_buys_top_scored_tickers_market_wide():
    store = MarketStore()
    _seed(store, "AAA", 100.0, "Tech")
    _seed(store, "BBB", 200.0, "Energy")
    _seed(store, "CCC", 50.0, "Finance")

    # AAA has the best score; CCC second; BBB worst.
    scores = {"AAA": 0.05, "CCC": 0.02, "BBB": -0.01}
    reactor, _ = _make_reactor(scores)

    pf = PortfolioView(cash=Decimal("50000"))
    intents = reactor.react(
        {"event_type": "BULL_RUN", "scope": "MARKET"},
        store, pf,
    )
    assert len(intents) >= 2
    assert all(i.side == "BUY" for i in intents)
    assert all(i.order_type == "MARKET" for i in intents)
    tickers_in_order = [i.ticker for i in intents]
    # Top score should be first
    assert tickers_in_order[0] == "AAA"


def test_sector_boom_only_buys_within_sector():
    store = MarketStore()
    _seed(store, "AAA", 100.0, "Tech")
    _seed(store, "TCH", 150.0, "Tech")
    _seed(store, "ENG", 200.0, "Energy")

    scores = {"AAA": 0.04, "TCH": 0.06, "ENG": 0.10}  # ENG would win market-wide
    reactor, _ = _make_reactor(scores)

    pf = PortfolioView(cash=Decimal("50000"))
    intents = reactor.react(
        {"event_type": "SECTOR_BOOM", "scope": "SECTOR", "target": "Tech"},
        store, pf,
    )
    tickers = {i.ticker for i in intents}
    assert "ENG" not in tickers
    assert tickers.issubset({"AAA", "TCH"})


# --------------------------------------------------------------------- SHOCK

def test_stock_shock_exits_only_named_held_ticker():
    store = MarketStore()
    _seed(store, "AAA", 100.0, "Tech")
    _seed(store, "BBB", 200.0, "Energy")

    pf = PortfolioView(
        cash=Decimal("0"),
        positions={
            "AAA": Position(ticker="AAA", quantity=50, avg_cost=Decimal("100"), peak_price=Decimal("100")),
            # BBB not held
        },
    )
    reactor, _ = _make_reactor()
    intents = reactor.react(
        {"event_type": "STOCK_SHOCK", "scope": "STOCK", "target": "AAA"},
        store, pf,
    )
    assert len(intents) == 1
    assert intents[0].ticker == "AAA"
    assert intents[0].side == "SELL"


def test_stock_shock_does_nothing_when_target_not_held():
    store = MarketStore()
    _seed(store, "XYZ", 100.0, "Tech")
    pf = PortfolioView(cash=Decimal("0"))
    reactor, _ = _make_reactor()
    intents = reactor.react(
        {"event_type": "STOCK_SHOCK", "scope": "STOCK", "target": "XYZ"},
        store, pf,
    )
    assert intents == []


def test_unknown_event_type_returns_no_intents():
    store = MarketStore()
    _seed(store, "AAA", 100.0, "Tech")
    pf = PortfolioView(cash=Decimal("10000"))
    reactor, _ = _make_reactor()
    intents = reactor.react({"event_type": "QUANTUM_QUAKE"}, store, pf)
    assert intents == []
