"""
Configuration loaded from environment variables (via .env in development).

Zero user-facing knobs by design — the bot self-tunes from market state.
The only things in .env are connection info, credentials, and the seed
deposit amount.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root if present (silent if not).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


def _optional(name: str, default: str) -> str:
    return os.getenv(name) or default


@dataclass(frozen=True)
class Settings:
    # Connection
    gateway_http_url: str
    gateway_ws_url: str

    # Bot account
    bot_email: str
    bot_password: str
    bot_username: str

    # Initial capital — the bot deposits this on boot if its balance is below it
    seed_deposit: Decimal
    seed_currency: str

    # Logging
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            gateway_http_url=_optional("GATEWAY_HTTP_URL", "http://localhost:8180").rstrip("/"),
            gateway_ws_url=_optional("GATEWAY_WS_URL", "ws://localhost:8180/notifications/ws"),
            bot_email=_required("BOT_EMAIL"),
            bot_password=_required("BOT_PASSWORD"),
            bot_username=_optional("BOT_USERNAME", "team2-bot"),
            seed_deposit=Decimal(_optional("SEED_DEPOSIT", "100000")),
            seed_currency=_optional("SEED_CURRENCY", "USD"),
            log_level=_optional("LOG_LEVEL", "INFO").upper(),
        )
