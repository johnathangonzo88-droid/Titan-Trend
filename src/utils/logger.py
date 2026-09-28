"""
Centralized logging setup.

Every module calls `get_logger(__name__)` instead of configuring its own
handlers, so log format/level/output are controlled in exactly one place.
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.utils.config_loader import PROJECT_ROOT

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def _configure_root(log_dir: str = "logs", level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_path = PROJECT_ROOT / log_dir
    log_path.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = RotatingFileHandler(
        log_path / "cronus_bot.log", maxBytes=5_000_000, backupCount=5
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str, log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """Return a module-level logger, configuring root handlers on first call."""
    _configure_root(log_dir=log_dir, level=level)
    return logging.getLogger(name)
