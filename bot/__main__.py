"""Entry point: `python -m bot [live | replay <file>]`."""
from __future__ import annotations

import asyncio
import sys

from .app import main, replay_main


if __name__ == "__main__":
    args = sys.argv[1:]

    if args and args[0] == "replay":
        if len(args) < 2:
            print("Usage: python -m bot replay <runs/timestamp.jsonl>")
            sys.exit(1)
        replay_main(args[1])
    else:
        # Default: run the live bot.
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            pass
