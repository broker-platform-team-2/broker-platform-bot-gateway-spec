"""
Streamlit live dashboard for trading bot monitoring.

Run with:
    streamlit run dashboards/app.py

Reads the most recent JSONL file from runs/ (written when REPLAY_RECORD=1).
Parses PRICE_UPDATE, ORDER_UPDATE (fills), and MARKET_EVENT messages.
Auto-refreshes every 5 seconds.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

RUNS_DIR = Path("runs")
REFRESH_SECONDS = 5
SEED_CASH = Decimal("100000")


# -------------------------------------------------------------------- helpers

def latest_run_file() -> Path | None:
    if not RUNS_DIR.exists():
        return None
    files = sorted(RUNS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def parse_run(path: Path) -> dict[str, Any]:
    """Parse a JSONL run file into structured data for the dashboard."""
    cash = SEED_CASH
    positions: dict[str, dict] = {}
    equity_curve: list[dict] = []
    fills: list[dict] = []
    events: list[dict] = []
    last_prices: dict[str, float] = {}
    price_series: dict[str, list[float]] = defaultdict(list)
    tick = 0

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
                ticker = payload.get("ticker") or payload.get("symbol")
                price = payload.get("price") or payload.get("current_price")
                if ticker and price:
                    last_prices[ticker] = float(price)
                    price_series[ticker].append(float(price))
                tick += 1
                if tick % 12 == 0:
                    pos_value = sum(
                        last_prices.get(t, p["avg_cost"]) * p["qty"]
                        for t, p in positions.items()
                    )
                    equity_curve.append({"tick": tick, "equity": float(cash) + pos_value})

            elif msg_type == "ORDER_UPDATE":
                status = payload.get("status")
                if status in ("FILLED", "PARTIALLY_FILLED"):
                    side = payload.get("side")
                    ticker = (
                        payload.get("instrumentId")
                        or payload.get("ticker")
                        or payload.get("symbol")
                    )
                    qty = int(
                        payload.get("filledQuantity")
                        or payload.get("quantity")
                        or 0
                    )
                    raw_price = (
                        payload.get("averagePrice")
                        or payload.get("limitPrice")
                        or last_prices.get(ticker, 0)
                    )
                    try:
                        price_d = Decimal(str(raw_price))
                    except InvalidOperation:
                        price_d = Decimal("0")

                    if not ticker or not side or qty <= 0:
                        continue

                    reason = payload.get("reason", "")
                    fills.append({
                        "tick": tick,
                        "side": side,
                        "ticker": ticker,
                        "qty": qty,
                        "price": float(price_d),
                        "reason": reason,
                    })

                    if side == "BUY":
                        cost = price_d * qty
                        if cost <= cash:
                            cash -= cost
                            if ticker in positions:
                                pos = positions[ticker]
                                total_qty = pos["qty"] + qty
                                pos["avg_cost"] = float(
                                    (Decimal(str(pos["avg_cost"])) * pos["qty"] + cost) / total_qty
                                )
                                pos["qty"] = total_qty
                            else:
                                positions[ticker] = {"qty": qty, "avg_cost": float(price_d)}
                    elif side == "SELL":
                        if ticker in positions:
                            pos = positions[ticker]
                            sell_qty = min(qty, pos["qty"])
                            cash += price_d * sell_qty
                            if sell_qty >= pos["qty"]:
                                del positions[ticker]
                            else:
                                positions[ticker]["qty"] -= sell_qty

            elif msg_type == "MARKET_EVENT":
                events.append({
                    "tick": tick,
                    "event_type": payload.get("event_type") or payload.get("eventType", ""),
                    "scope": payload.get("scope", ""),
                    "target": payload.get("target", ""),
                    "headline": payload.get("headline", ""),
                    "magnitude": payload.get("magnitude", ""),
                })

    pos_value = sum(
        last_prices.get(t, p["avg_cost"]) * p["qty"]
        for t, p in positions.items()
    )
    final_equity = float(cash) + pos_value
    pnl = final_equity - float(SEED_CASH)
    pnl_pct = (pnl / float(SEED_CASH)) * 100

    buys = [f for f in fills if f["side"] == "BUY"]
    sells = [f for f in fills if f["side"] == "SELL"]
    wins = sum(1 for f in sells if f["price"] > 0)

    return {
        "equity_curve": equity_curve,
        "positions": positions,
        "last_prices": last_prices,
        "price_series": dict(price_series),
        "fills": fills,
        "events": events,
        "final_equity": final_equity,
        "cash": float(cash),
        "ticks": tick,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "buy_count": len(buys),
        "sell_count": len(sells),
        "win_count": wins,
    }


# -------------------------------------------------------------------- layout

st.set_page_config(
    page_title="Bot Dashboard",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="collapsed",
)

st.title("📈 Trading Bot — Live Dashboard")

run_file = latest_run_file()

if run_file is None:
    st.warning(
        "No run files found in `runs/`. Start the bot with `REPLAY_RECORD=1` to enable recording."
    )
    st.stop()

file_age_s = int(time.time() - run_file.stat().st_mtime)
age_label = f"{file_age_s}s ago" if file_age_s < 120 else f"{file_age_s // 60}m ago"
st.caption(f"File: `{run_file.name}` — last modified {age_label} — refreshing every {REFRESH_SECONDS}s")

data = parse_run(run_file)

# ── row 1: key metrics ────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

pnl_delta = f"${data['pnl']:+,.2f} ({data['pnl_pct']:+.2f}%)"
c1.metric("Equity", f"${data['final_equity']:,.2f}", delta=pnl_delta,
          delta_color="normal")
c2.metric("Cash", f"${data['cash']:,.2f}")
c3.metric("Open Positions", len(data["positions"]))
c4.metric("Fills (B/S)", f"{data['buy_count']} / {data['sell_count']}")
c5.metric("Ticks", f"{data['ticks']:,}")

st.divider()

# ── row 2: equity curve + PnL per ticker ─────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.subheader("Equity Curve")
    if data["equity_curve"]:
        df_eq = pd.DataFrame(data["equity_curve"]).set_index("tick")
        df_eq["seed"] = float(SEED_CASH)
        st.line_chart(df_eq[["equity", "seed"]], use_container_width=True,
                      color=["#26a641", "#555555"])
    else:
        st.info("Waiting for price ticks…")

with right:
    st.subheader("P&L per Ticker")
    if data["positions"]:
        rows = []
        for ticker, pos in data["positions"].items():
            last_px = data["last_prices"].get(ticker, pos["avg_cost"])
            pnl_pct_t = (last_px / pos["avg_cost"] - 1) * 100 if pos["avg_cost"] else 0
            rows.append({"ticker": ticker, "pnl_pct": round(pnl_pct_t, 2)})
        df_pnl = pd.DataFrame(rows).sort_values("pnl_pct")
        st.bar_chart(df_pnl.set_index("ticker")["pnl_pct"], use_container_width=True,
                     color="#26a641")
    else:
        st.info("No open positions.")

st.divider()

# ── row 3: positions table + risk metrics ─────────────────────────────────────
left2, right2 = st.columns([3, 2])

with left2:
    st.subheader("Open Positions")
    if data["positions"]:
        rows = []
        for ticker, pos in data["positions"].items():
            last_px = data["last_prices"].get(ticker, pos["avg_cost"])
            pnl_pct_t = (last_px / pos["avg_cost"] - 1) * 100 if pos["avg_cost"] else 0
            notional = last_px * pos["qty"]
            rows.append({
                "Ticker": ticker,
                "Qty": pos["qty"],
                "Avg Cost": pos["avg_cost"],
                "Last Price": last_px,
                "P&L %": pnl_pct_t,
                "Notional": notional,
            })
        df_pos = pd.DataFrame(rows)

        def _color_pnl(val: float) -> str:
            if val > 0:
                return "color: #26a641; font-weight: bold"
            if val < 0:
                return "color: #da3633; font-weight: bold"
            return ""

        styled = (
            df_pos.style
            .format({
                "Avg Cost": "${:.2f}",
                "Last Price": "${:.2f}",
                "P&L %": "{:+.2f}%",
                "Notional": "${:,.0f}",
            })
            .applymap(_color_pnl, subset=["P&L %"])
        )
        st.dataframe(styled, hide_index=True, use_container_width=True)
    else:
        st.info("No open positions.")

with right2:
    st.subheader("Risk Snapshot")
    if data["equity_curve"]:
        equities = [e["equity"] for e in data["equity_curve"]]
        peak_eq = max(equities)
        drawdown_pct = ((data["final_equity"] - peak_eq) / peak_eq * 100) if peak_eq else 0
        total_notional = sum(
            data["last_prices"].get(t, p["avg_cost"]) * p["qty"]
            for t, p in data["positions"].items()
        )
        exposure_pct = (total_notional / data["final_equity"] * 100) if data["final_equity"] else 0

        dd_color = "inverse" if drawdown_pct < -5 else "normal"
        st.metric("Drawdown from Peak", f"{drawdown_pct:+.2f}%", delta_color=dd_color)
        st.metric("Portfolio Exposure", f"{exposure_pct:.1f}%")
        st.metric("Peak Equity", f"${peak_eq:,.2f}")

        if drawdown_pct <= -15:
            st.error("⚠️ Kill switch threshold reached (-15%)")
        elif drawdown_pct <= -10:
            st.warning("⚠️ Approaching kill switch threshold")
    else:
        st.info("Waiting for data…")

st.divider()

# ── row 4: recent fills + trade reason breakdown ──────────────────────────────
left3, right3 = st.columns([3, 2])

with left3:
    st.subheader("Recent Fills (last 25)")
    if data["fills"]:
        df_fills = pd.DataFrame(data["fills"][-25:]).copy()
        df_fills = df_fills[["tick", "side", "ticker", "qty", "price", "reason"]]

        def _color_side(val: str) -> str:
            return "color: #26a641; font-weight: bold" if val == "BUY" else "color: #da3633; font-weight: bold"

        styled_fills = (
            df_fills.style
            .format({"price": "${:.2f}"})
            .applymap(_color_side, subset=["side"])
        )
        st.dataframe(styled_fills, hide_index=True, use_container_width=True)
    else:
        st.info("No fills recorded yet.")

with right3:
    st.subheader("Exit Reasons")
    sell_fills = [f for f in data["fills"] if f["side"] == "SELL" and f.get("reason")]
    if sell_fills:
        reason_counts = defaultdict(int)
        for f in sell_fills:
            reason_counts[f["reason"]] += 1
        df_reasons = pd.DataFrame(
            list(reason_counts.items()), columns=["reason", "count"]
        ).sort_values("count", ascending=False)
        st.bar_chart(df_reasons.set_index("reason")["count"], use_container_width=True)
    else:
        st.info("No sell fills yet.")

st.divider()

# ── row 5: market events ──────────────────────────────────────────────────────
st.subheader("Market Events")
if data["events"]:
    df_events = pd.DataFrame(data["events"])
    df_events = df_events[["tick", "event_type", "scope", "target", "magnitude", "headline"]]

    _event_colors = {
        "BULL_RUN": "background-color: #0d3b0d",
        "SECTOR_BOOM": "background-color: #0d2b0d",
        "BEAR_CRASH": "background-color: #3b0d0d",
        "SECTOR_SLUMP": "background-color: #2b0d0d",
        "STOCK_SHOCK": "background-color: #2b2b0d",
    }

    def _color_event_row(row: pd.Series) -> list[str]:
        style = _event_colors.get(row["event_type"], "")
        return [style] * len(row)

    styled_events = df_events.style.apply(_color_event_row, axis=1)
    st.dataframe(styled_events, hide_index=True, use_container_width=True)
else:
    st.info("No market events received yet.")

# ── auto-refresh ──────────────────────────────────────────────────────────────
time.sleep(REFRESH_SECONDS)
st.rerun()
