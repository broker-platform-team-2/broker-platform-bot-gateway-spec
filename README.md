# Trading Bot

Autonomous, aggressive, zero-config trading bot for the Spring Practice 2026
Stock Exchange Application competition. Talks to our team's broker platform
(`user-gateway` on port 8180) and is driven by a single decision pipeline:
scan → score → allocate → pyramid → exit.

See `trading_bot_implementation_plan_v3.docx` (in the `broker-platform` repo)
for the full design.

## Quick start

```
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
cp .env.example .env              # fill in BOT_EMAIL and BOT_PASSWORD
python -m bot
```

## Project layout

```
bot/
  __init__.py
  __main__.py        # CLI entry: `python -m bot`
  config.py          # constants + .env reader
  http_client.py     # httpx wrapper with JWT auth
  ws_client.py       # /notifications/ws consumer
  market.py          # price ring buffer + portfolio cache
  indicators.py      # numpy: ATR, EMA, donchian, etc.
  strategy.py        # scanner + entry/exit/pyramid logic
  events.py          # event listener + reactor
  executor.py        # order builder + retry
  risk.py            # drawdown kill-switch + rate budget
  app.py             # the tick loop
tests/
  __init__.py
```
