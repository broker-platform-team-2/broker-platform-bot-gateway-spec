# Trading Bot — Spring Practice 2026

Autonomous, aggressive, zero-config trading bot for the Spring Practice 2026
Stock Exchange Application competition. Talks to the broker platform
(`user-gateway` on port 8180). Single decision pipeline:
**scan → score → allocate → pyramid → react to events → exit**.

Built in 4 days (Track A: trading smarts · Track B: robustness & visibility).

---

## Quick start

```powershell
# 1. Create and activate the virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1       # Windows PowerShell
# source .venv/bin/activate        # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
Copy-Item .env.example .env        # then edit .env: set BOT_EMAIL + BOT_PASSWORD

# 4. Run
python -m bot
```

The bot registers its own account on first run, deposits seed capital, and
starts trading — no manual steps needed.

---

## Simulate offline (no broker required)

Run a full realistic scenario against generated price data. Exercises every
feature: momentum entries, pyramiding, event reactions, trailing stops,
hard stops, risk gate, replay recording.

```powershell
python scripts/simulate.py
```

Sample output:
```
=== Offline Simulation — Spring Practice 2026 Bot ===
  BUY   NVDA   qty=312   @491.43  [entry]       equity=$100,000  tick=372
  BUY   AAPL   qty=870   @173.08  [entry]       equity=$99,891   tick=372
  EVENT tick= 350  BULL_RUN  "Broad market rally — all sectors surging!"
  BUY   NVDA   qty=208   @510.22  [event_bull]  equity=$98,203   tick=350
  ...
  SELL  BAC    qty=500   @33.10   [hard_stop]   equity=$96,200   tick=490
  ...
  P&L  +$18,432 (+18.43%)
```

After the simulation finishes, replay it or open the dashboard:

```powershell
python -m bot replay runs\sim_<timestamp>.jsonl
streamlit run dashboards\app.py      # open http://localhost:8501
```

---

## Running with the live broker

### Normal mode

```powershell
python -m bot
```

### With replay recording (enables dashboard + post-mortem replay)

```powershell
$env:REPLAY_RECORD = "1"
python -m bot
```

Every WebSocket message and order outcome is written to `runs/<timestamp>.jsonl`.

### Watch the dashboard during a live run

In a second terminal (while the bot is running with `REPLAY_RECORD=1`):

```powershell
streamlit run dashboards\app.py
```

Opens at `http://localhost:8501`. Auto-refreshes every 5 seconds. Shows:
- Equity curve
- Open positions with unrealised P&L
- Last 20 fills
- Top-10 tickers by last price

### Replay a past run

```powershell
python -m bot replay runs\<timestamp>.jsonl
```

Replays the run through the live strategy deterministically and prints a PnL
summary + trade list.

---

## Tests

```powershell
python -m pytest tests/ -v
```

49 tests covering indicators, strategy, pyramiding, events, risk gate,
executor (reconcile-then-retry), order book cache, and replay harness.

---

## Project layout

```
bot/
  __main__.py        # CLI entry: `python -m bot` or `python -m bot replay <file>`
  config.py          # constants + .env reader (zero user-facing knobs)
  http_client.py     # async httpx wrapper — JWT auth, retry, reconcile helper
  ws_client.py       # /notifications/ws consumer — auto-reconnect + recorder hook
  market.py          # price + volume ring buffers per ticker, sector routing
  indicators.py      # numpy: ATR, EMA, Donchian, momentum_score, volume_surge
  strategy.py        # scanner, entry/exit, pyramiding (Track A)
  events.py          # BULL_RUN / BEAR_CRASH / SECTOR_BOOM / SLUMP / SHOCK (Track A)
  executor.py        # order builder — reconcile-then-retry on timeout/5xx (Track B)
  orderbook.py       # top-of-book cache fed by ORDER_BOOK_UPDATE (Track B)
  replay.py          # JSONL recorder + offline replay engine (Track B)
  risk.py            # drawdown kill-switch, rate budget, position caps
  app.py             # main event loop — wires everything together

scripts/
  simulate.py        # offline simulation — full scenario, no broker needed

dashboards/
  app.py             # Streamlit dashboard (run with `streamlit run dashboards/app.py`)

tests/
  test_events.py
  test_executor.py
  test_indicators.py
  test_orderbook.py
  test_replay.py
  test_risk.py
  test_smoke.py
  test_strategy.py
```

---

## Environment variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `GATEWAY_HTTP_URL` | `http://localhost:8180` | Broker platform HTTP base URL |
| `GATEWAY_WS_URL` | `ws://localhost:8180/notifications/ws` | WebSocket endpoint |
| `BOT_EMAIL` | *(required)* | Bot account email |
| `BOT_PASSWORD` | *(required)* | Bot account password |
| `BOT_USERNAME` | `team2-bot` | Display name (used on first register) |
| `SEED_DEPOSIT` | `100000` | Capital deposited on boot if balance is below this |
| `SEED_CURRENCY` | `USD` | Currency for seed deposit |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |
| `REPLAY_RECORD` | *(unset)* | Set to `1` to record all WS messages to `runs/` |
