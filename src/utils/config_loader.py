"""
Configuration loading utilities.

Centralizes reading of `config/config.yaml` and `.env` so every module gets
its settings the same way instead of scattering `open()` calls throughout
the codebase.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

# Resolve the project root regardless of the caller's current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


@dataclass
class Settings:
    """Typed wrapper around environment secrets (never logged, never committed)."""

    telegram_bot_token: str
    telegram_chat_ids: list
    live_data_source: str
    broker_api_key: str
    broker_api_secret: str
    env: str

    @classmethod
    def from_env(cls, env_path: Path = DEFAULT_ENV_PATH) -> "Settings":
        if env_path.exists():
            load_dotenv(env_path)
        chat_ids_raw = os.getenv("TELEGRAM_CHAT_IDS", "")
        chat_ids = [c.strip() for c in chat_ids_raw.split(",") if c.strip()]
        return cls(
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_ids=chat_ids,
            live_data_source=os.getenv("LIVE_DATA_SOURCE", "csv"),
            broker_api_key=os.getenv("BROKER_API_KEY", ""),
            broker_api_secret=os.getenv("BROKER_API_SECRET", ""),
            env=os.getenv("ENV", "development"),
        )


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load and return the YAML config as a plain dict.

    Raises FileNotFoundError with an explicit path if the config is missing,
    rather than letting yaml raise a confusing generic error.
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str) -> Path:
    """Resolve a path from config (e.g. 'data/raw/ES_15m.csv') against project root."""
    p = Path(relative)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p
