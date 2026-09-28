#!/usr/bin/env python3
"""
Live signal generation script: the entry point a scheduler (cron, GitHub
Actions, a systemd timer, or the Docker container's own loop) invokes to
check every configured symbol for a new signal and broadcast it to Telegram.

Usage:
    python scripts/live_signal.py --once       # single check, then exit (for cron/CI)
    python scripts/live_signal.py --forever    # long-running loop (for Docker/VPS)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.telegram_bot.bot import SignalBot
from src.utils.config_loader import Settings, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and broadcast live trading signals.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--once", action="store_true", help="Run a single check and exit (default)")
    parser.add_argument("--forever", action="store_true", help="Run continuously, polling on an interval")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    settings = Settings.from_env()

    if not settings.telegram_bot_token:
        logger.error("TELEGRAM_BOT_TOKEN is not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)
    if not settings.telegram_chat_ids:
        logger.error("TELEGRAM_CHAT_IDS is not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    bot = SignalBot(config=config, settings=settings)

    if args.forever:
        bot.run_forever()
    else:
        sent = bot.run_once()
        logger.info("Signal check complete. %d signal(s) sent.", len(sent))


if __name__ == "__main__":
    main()
