"""
Minimal Telegram Bot API client using plain HTTP (`requests`).

Deliberately does not depend on the `python-telegram-bot` package: sending
signal messages only needs two Bot API endpoints (`sendMessage`,
`getUpdates`), and a thin HTTP wrapper is easier to test, has zero extra
dependency surface, and avoids version churn in a heavier framework. If you
later want interactive commands (inline buttons, `/subscribe` flows), the
`python-telegram-bot` dependency is already listed in requirements.txt and
can be layered on top of - or instead of - this client.
"""
from __future__ import annotations

import time
from typing import Optional

import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

_API_BASE = "https://api.telegram.org/bot{token}/{method}"


class TelegramClient:
    def __init__(self, bot_token: str, parse_mode: str = "HTML", timeout: int = 15):
        if not bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is empty - set it in your .env file")
        self.bot_token = bot_token
        self.parse_mode = parse_mode
        self.timeout = timeout

    def _url(self, method: str) -> str:
        return _API_BASE.format(token=self.bot_token, method=method)

    def send_message(self, chat_id: str, text: str, retries: int = 3, backoff_seconds: float = 2.0) -> bool:
        """Send one message, retrying on transient network/HTTP errors.
        Returns True on success, False if all retries are exhausted (never
        raises, so one bad chat_id doesn't take down a multi-subscriber broadcast).
        """
        payload = {"chat_id": chat_id, "text": text, "parse_mode": self.parse_mode}
        for attempt in range(1, retries + 1):
            try:
                resp = requests.post(self._url("sendMessage"), json=payload, timeout=self.timeout)
                if resp.status_code == 200 and resp.json().get("ok"):
                    return True
                logger.error(
                    "Telegram sendMessage failed (attempt %d/%d) chat_id=%s status=%s body=%s",
                    attempt, retries, chat_id, resp.status_code, resp.text[:300],
                )
            except requests.RequestException as e:
                logger.error("Telegram sendMessage network error (attempt %d/%d): %s", attempt, retries, e)
            if attempt < retries:
                time.sleep(backoff_seconds * attempt)
        return False

    def broadcast(self, chat_ids: list, text: str) -> dict:
        """Send the same message to every configured chat/channel. Returns
        {chat_id: success_bool} so callers can log/alert on partial failures.
        """
        results = {}
        for chat_id in chat_ids:
            results[chat_id] = self.send_message(chat_id, text)
        n_ok = sum(results.values())
        logger.info("Broadcast delivered to %d/%d chats", n_ok, len(chat_ids))
        return results

    def get_updates(self, offset: Optional[int] = None, timeout: int = 30) -> list:
        """Long-poll for new updates (e.g. subscribers sending /start).
        Used by the optional subscriber-management loop, not required for
        one-way signal broadcasting to a fixed channel/group ID.
        """
        params = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset
        try:
            resp = requests.get(self._url("getUpdates"), params=params, timeout=timeout + 10)
            resp.raise_for_status()
            return resp.json().get("result", [])
        except requests.RequestException as e:
            logger.error("Telegram getUpdates failed: %s", e)
            return []
