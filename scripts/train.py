#!/usr/bin/env python3
"""
Training script: run the full ML pipeline (feature engineering -> leak-free
labeling -> walk-forward validation across all available models -> best
model selection -> refit and persist) for every `ml_enabled: true` symbol in
config.yaml.

Usage:
    python scripts/train.py
    python scripts/train.py --config config/config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.train import train_all
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML models for all ML-enabled symbols.")
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: config/config.yaml)")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()

    logger.info("Starting training run for symbols: %s", [s["name"] for s in config["symbols"] if s.get("ml_enabled")])
    summaries = train_all(config)

    print(json.dumps(summaries, indent=2, default=str))

    failed = [s for s, r in summaries.items() if r.get("status") != "ok"]
    if failed:
        logger.error("Training failed for: %s", failed)
        sys.exit(1)
    logger.info("Training complete for all symbols.")


if __name__ == "__main__":
    main()
