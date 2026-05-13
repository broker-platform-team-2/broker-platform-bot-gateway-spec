"""
Replay harness.

Recording mode (REPLAY_RECORD=1 env var):
  Every WebSocket message received is written as a JSONL line to
  runs/<unix-timestamp>.jsonl. Order outcomes are appended too.

Replay mode (`python -m bot replay runs/<file>.jsonl`):
  Reads the JSONL file, drives PRICE_UPDATE messages through the strategy
  deterministically, prints a PnL summary and the last 10 trades.
"""
from __future__ import annotations

import json
import os
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

RUNS_DIR = Path("runs")

ONE_TICK = Decimal("0.01")


def recording_enabled() -> bool:
    return os.environ.get("REPLAY_RECORD", "").strip() == "1"


def open_run_file() -> "RunRecorder":
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = RUNS_DIR / f"{ts}.jsonl"
    return RunRecorder(path)


class RunRecorder:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh = path.open("a", encoding="utf-8")

    def record_ws(self, msg_type: str, payload: dict[str, Any]) -> None:
        line = json.dumps({"kind": "ws", "type": msg_type, "payload": payload})
        self._fh.write(line + "\n")
        self._fh.flush()

    def record_order(
        self,
        *,
        side: str,
        ticker: str,
        qty: int,
        order_type: str,
        limit_price: Decimal | None,
        reason: str,
        result: dict[str, Any] | None,
    ) -> None:
        data: dict[str, Any] = {
            "kind": "order",
            "side": side,
            "ticker": ticker,
            "qty": qty,
            "order_type": order_type,
            "limit_price": str(limit_price) if limit_price else None,
            "reason": reason,
            "result": result,
        }
        self._fh.write(json.dumps(data) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    @property
    def path(self) -> Path:
        return self._path


# ------------------------------------------------------------------- replay CLI

def replay(jsonl_path: str, seed_cash: Decimal = Decimal("100000")) -> None:
    """Replay a recorded run and print a PnL summary."""
    from .market import MarketStore
    from .strategy import OrderIntent, PortfolioView, Position, Strategy

    path = Path(jsonl_path)
    if not path.exists():
        print(f"File not found: {path}")
        return

    market = MarketStore()
    strategy = Strategy()
    portfolio = PortfolioView(cash=seed_cash)
    trades: list[dict[str, Any]] = []
    tick = 0

    DECISION_INTERVAL = 12

    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = entry.get("kind")
            if kind != "ws":
                continue
            msg_type = entry.get("type")
            payload = entry.get("payload", {})

            if msg_type == "PRICE_UPDATE":
                market.apply_price_update(payload)
                tick += 1
                if tick % DECISION_INTERVAL != 0:
                    continue

                intents = strategy.decide(market, portfolio)
                for intent in intents:
                    state = market.get(intent.ticker)
                    if not state:
                        continue
                    price = Decimal(str(state.price))
                    trades.append(
                        {
                            "tick": tick,
                            "side": intent.side,
                            "ticker": intent.ticker,
                            "qty": intent.quantity,
                            "price": price,
                            "reason": intent.reason,
                        }
                    )
                    _apply_fill(portfolio, intent, price)

    final_equity = portfolio.equity(market)
    gain = final_equity - seed_cash
    gain_pct = (gain / seed_cash * 100) if seed_cash else Decimal("0")

    print(f"\n=== Replay: {path.name} ===")
    print(f"Ticks processed : {tick}")
    print(f"Trades simulated: {len(trades)}")
    print(f"Final equity    : ${final_equity:,.2f}")
    print(f"P&L             : ${gain:+,.2f} ({gain_pct:+.2f}%)")
    if trades:
        print("\nLast 10 trades:")
        for t in trades[-10:]:
            print(
                f"  tick={t['tick']:>5}  {t['side']:<4}  {t['ticker']:<8}"
                f"  qty={t['qty']:<6}  @{t['price']:.2f}  [{t['reason']}]"
            )


def _apply_fill(
    portfolio: Any,
    intent: Any,
    price: Decimal,
) -> None:
    """Simulate an immediate fill at the given price (replay only)."""
    from .strategy import Position

    if intent.side == "BUY":
        cost = price * intent.quantity
        if cost > portfolio.cash:
            return
        portfolio.cash -= cost
        pos = portfolio.positions.get(intent.ticker)
        if pos:
            total_qty = pos.quantity + intent.quantity
            new_avg = (pos.avg_cost * pos.quantity + cost) / total_qty
            pos.quantity = total_qty
            pos.avg_cost = new_avg
            pos.peak_price = max(pos.peak_price, price)
        else:
            portfolio.positions[intent.ticker] = Position(
                ticker=intent.ticker,
                quantity=intent.quantity,
                avg_cost=price,
                peak_price=price,
            )
    elif intent.side == "SELL":
        pos = portfolio.positions.get(intent.ticker)
        if pos:
            qty = min(intent.quantity, pos.quantity)
            portfolio.cash += price * qty
            if qty >= pos.quantity:
                del portfolio.positions[intent.ticker]
            else:
                pos.quantity -= qty
