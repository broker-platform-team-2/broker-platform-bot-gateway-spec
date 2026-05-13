"""Tests for the replay harness."""
from __future__ import annotations

import json
import tempfile
from decimal import Decimal
from pathlib import Path

import pytest

from bot.replay import RunRecorder, replay


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")


def test_run_recorder_writes_ws_messages(tmp_path):
    rec_path = tmp_path / "test.jsonl"
    rec = RunRecorder(rec_path)
    rec.record_ws("PRICE_UPDATE", {"ticker": "ARKA", "price": 100.0})
    rec.record_ws("MARKET_EVENT", {"event_type": "BULL_RUN"})
    rec.close()

    lines = [json.loads(l) for l in rec_path.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0] == {"kind": "ws", "type": "PRICE_UPDATE", "payload": {"ticker": "ARKA", "price": 100.0}}
    assert lines[1]["type"] == "MARKET_EVENT"


def test_run_recorder_writes_orders(tmp_path):
    rec_path = tmp_path / "test.jsonl"
    rec = RunRecorder(rec_path)
    rec.record_order(
        side="BUY",
        ticker="ARKA",
        qty=10,
        order_type="LIMIT",
        limit_price=Decimal("100.10"),
        reason="entry",
        result={"exchangeOrderId": "ORD-1"},
    )
    rec.close()

    lines = [json.loads(l) for l in rec_path.read_text().splitlines()]
    assert lines[0]["kind"] == "order"
    assert lines[0]["ticker"] == "ARKA"
    assert lines[0]["limit_price"] == "100.10"


def test_replay_empty_file(tmp_path, capsys):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    replay(str(p), seed_cash=Decimal("10000"))
    out = capsys.readouterr().out
    assert "Ticks processed" in out
    assert "0" in out


def test_replay_produces_pnl_summary(tmp_path, capsys):
    p = tmp_path / "run.jsonl"
    # 40 PRICE_UPDATEs (enough for scoring) + steady uptrend across 5 tickers
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    lines = []
    for i in range(40):
        for t, base in zip(tickers, [100, 200, 50, 80, 60]):
            lines.append(
                {"kind": "ws", "type": "PRICE_UPDATE",
                 "payload": {"ticker": t, "price": base + 0.5 * i, "volume": 1000.0}}
            )
    _write_jsonl(p, lines)

    replay(str(p), seed_cash=Decimal("100000"))
    out = capsys.readouterr().out
    assert "Replay:" in out
    assert "Final equity" in out
    assert "P&L" in out


def test_replay_file_not_found(tmp_path, capsys):
    replay(str(tmp_path / "missing.jsonl"))
    out = capsys.readouterr().out
    assert "not found" in out.lower()
