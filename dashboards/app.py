"""
Streamlit dashboard for live bot monitoring.

Run with:
    streamlit run dashboards/app.py

Reads the most recent JSONL file from runs/ (written by the bot when
REPLAY_RECORD=1). Auto-refreshes every 5 seconds so the charts stay current
during a live run.

Panels:
  - Equity curve (reconstructed from fills in the JSONL)
  - Current open positions (last SELL/BUY pair per ticker)
  - Last 20 fills
  - Top-10 momentum candidates (most recent PRICE_UPDATE per ticker)
"""
from __future__ import annotations

import json
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

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


def parse_run(path: Path) -> dict:
    """Parse a JSONL run file into structured data."""
    cash = SEED_CASH
    positions: dict[str, dict] = {}   # ticker -> {qty, avg_cost}
    equity_curve: list[dict] = []
    fills: list[dict] = []
    last_prices: dict[str, float] = {}
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
            if kind == "ws":
                msg_type = entry.get("type")
                payload = entry.get("payload", {})
                if msg_type == "PRICE_UPDATE":
                    ticker = payload.get("ticker") or payload.get("symbol")
                    price = payload.get("price") or payload.get("current_price")
                    if ticker and price:
                        last_prices[ticker] = float(price)
                    tick += 1
                    if tick % 12 == 0:
                        # Compute equity snapshot
                        pos_value = sum(
                            last_prices.get(t, p["avg_cost"]) * p["qty"]
                            for t, p in positions.items()
                        )
                        equity = float(cash) + pos_value
                        equity_curve.append({"tick": tick, "equity": equity})

            elif kind == "order":
                side = entry.get("side")
                ticker = entry.get("ticker")
                qty = entry.get("qty", 0)
                price_str = entry.get("limit_price")
                reason = entry.get("reason", "")
                result = entry.get("result")
                if not ticker or not side or result is None:
                    continue
                try:
                    price = Decimal(str(price_str)) if price_str else Decimal(str(last_prices.get(ticker, 0)))
                except InvalidOperation:
                    continue

                fills.append({
                    "tick": tick,
                    "side": side,
                    "ticker": ticker,
                    "qty": qty,
                    "price": float(price),
                    "reason": reason,
                })

                if side == "BUY":
                    cost = price * qty
                    if cost <= cash:
                        cash -= cost
                        if ticker in positions:
                            pos = positions[ticker]
                            total_qty = pos["qty"] + qty
                            pos["avg_cost"] = (Decimal(str(pos["avg_cost"])) * pos["qty"] + cost) / total_qty
                            pos["qty"] = total_qty
                        else:
                            positions[ticker] = {"qty": qty, "avg_cost": float(price)}
                elif side == "SELL":
                    if ticker in positions:
                        pos = positions[ticker]
                        sell_qty = min(qty, pos["qty"])
                        cash += price * sell_qty
                        if sell_qty >= pos["qty"]:
                            del positions[ticker]
                        else:
                            positions[ticker]["qty"] -= sell_qty

    # Final equity with latest prices
    pos_value = sum(
        last_prices.get(t, p["avg_cost"]) * p["qty"]
        for t, p in positions.items()
    )
    final_equity = float(cash) + pos_value

    return {
        "equity_curve": equity_curve,
        "positions": positions,
        "last_prices": last_prices,
        "fills": fills,
        "final_equity": final_equity,
        "cash": float(cash),
        "ticks": tick,
    }


# -------------------------------------------------------------------- layout

st.set_page_config(page_title="Bot Dashboard", layout="wide", page_icon="chart_with_upwards_trend")
st.title("Trading Bot - Live Dashboard")

run_file = latest_run_file()

if run_file is None:
    st.warning(
        "No run files found in `runs/`. Start the bot with `REPLAY_RECORD=1` "
        "to enable recording."
    )
    st.stop()

st.caption(f"Reading: `{run_file.name}` — auto-refresh every {REFRESH_SECONDS}s")

data = parse_run(run_file)

# ---- top metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Final Equity", f"${data['final_equity']:,.2f}")
col2.metric("Cash", f"${data['cash']:,.2f}")
col3.metric("Open Positions", len(data["positions"]))
col4.metric("Ticks Processed", data["ticks"])

st.divider()

# ---- equity curve
left, right = st.columns([2, 1])

with left:
    st.subheader("Equity Curve")
    if data["equity_curve"]:
        df_eq = pd.DataFrame(data["equity_curve"]).set_index("tick")
        st.line_chart(df_eq["equity"], use_container_width=True)
    else:
        st.info("No equity data yet.")

# ---- open positions
with right:
    st.subheader("Open Positions")
    if data["positions"]:
        rows = []
        for ticker, pos in data["positions"].items():
            last_px = data["last_prices"].get(ticker, pos["avg_cost"])
            pnl_pct = (last_px / pos["avg_cost"] - 1) * 100 if pos["avg_cost"] else 0
            rows.append({
                "Ticker": ticker,
                "Qty": pos["qty"],
                "Avg Cost": f"${pos['avg_cost']:.2f}",
                "Last": f"${last_px:.2f}",
                "P&L %": f"{pnl_pct:+.2f}%",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No open positions.")

st.divider()

# ---- last 20 fills
st.subheader("Last 20 Fills")
if data["fills"]:
    df_fills = pd.DataFrame(data["fills"][-20:])
    df_fills["price"] = df_fills["price"].map(lambda x: f"${x:.2f}")
    st.dataframe(df_fills, hide_index=True, use_container_width=True)
else:
    st.info("No fills recorded yet.")

# ---- top-10 momentum (by most recent price magnitude — a proxy)
st.subheader("Top-10 Tickers by Last Price")
if data["last_prices"]:
    top10 = sorted(data["last_prices"].items(), key=lambda kv: kv[1], reverse=True)[:10]
    df_top = pd.DataFrame(top10, columns=["Ticker", "Last Price"])
    df_top["Last Price"] = df_top["Last Price"].map(lambda x: f"${x:.2f}")
    st.dataframe(df_top, hide_index=True, use_container_width=True)
else:
    st.info("No price data yet.")

# Auto-refresh
st.markdown(
    f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}">',
    unsafe_allow_html=True,
)
