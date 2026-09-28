"""
End-to-end training orchestration for one symbol: build features -> build
labeled dataset -> walk-forward evaluate every available model -> pick the
best -> refit it on the full dataset for deployment -> persist to disk.

The walk-forward metrics (out-of-sample, never touching the final refit) are
what should be trusted for "how good is this model"; the full-data refit is
purely so live_signal.py has a single model file trained on all available
history to score the *next*, still-unseen bar.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.data_layer.loader import load_ohlcv_csv
from src.features.feature_engineering import build_features
from src.ml.dataset import build_ml_dataset
from src.ml.models import get_available_models
from src.ml.walkforward import run_walk_forward_evaluation, select_best_model
from src.utils.config_loader import resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_symbol(symbol_cfg: dict, features_cfg: dict, ml_cfg: dict, model_dir: str) -> dict:
    """Train and evaluate all models for one symbol. Returns a summary dict
    (also written to `<model_dir>/<symbol>_training_summary.json`).
    """
    symbol = symbol_cfg["name"]
    logger.info("=== Training %s ===", symbol)

    df, quality_report = load_ohlcv_csv(symbol_cfg["raw_file"], symbol)
    logger.info(quality_report.summary())

    features = build_features(df, features_cfg, has_volume=quality_report.has_volume)
    X, y, feature_cols = build_ml_dataset(features, ml_cfg)

    results = run_walk_forward_evaluation(X, y, ml_cfg)
    if not results:
        logger.error("No models could be trained for %s (no libraries available / insufficient data)", symbol)
        return {"symbol": symbol, "status": "failed", "reason": "no_models_available"}

    best_name = select_best_model(results, metric="roc_auc")

    # Refit the winning model on the FULL dataset for deployment. This is the
    # only place a model is allowed to see the entire history; it is never
    # used to produce the walk-forward metrics reported to the user.
    factories = get_available_models([best_name])
    final_model = factories[best_name]()
    final_model.fit(X.values, y.values)

    model_path = resolve_path(model_dir) / f"{symbol}_{best_name}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "feature_cols": feature_cols, "model_name": best_name}, model_path)
    logger.info("Saved deployed model for %s -> %s", symbol, model_path)

    summary = {
        "symbol": symbol,
        "status": "ok",
        "data_quality": {
            "n_rows": quality_report.n_rows,
            "start": str(quality_report.start),
            "end": str(quality_report.end),
            "has_volume": quality_report.has_volume,
            "notes": quality_report.notes,
        },
        "n_ml_rows": len(X),
        "label_balance": y.value_counts(normalize=True).to_dict(),
        "best_model": best_name,
        "model_path": str(model_path),
        "walk_forward_metrics": {
            name: {
                "mean": res.mean_metrics,
                "n_folds": len(res.fold_metrics),
            }
            for name, res in results.items()
        },
    }

    summary_path = resolve_path(model_dir) / f"{symbol}_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Training summary written to %s", summary_path)

    return summary


def train_all(config: dict) -> dict:
    """Train every symbol flagged `ml_enabled: true` in config.yaml."""
    all_summaries = {}
    for symbol_cfg in config["symbols"]:
        if not symbol_cfg.get("ml_enabled", False):
            logger.info("Skipping ML training for %s (ml_enabled: false)", symbol_cfg["name"])
            continue
        summary = train_symbol(
            symbol_cfg, config["features"], config["ml"], config["ml"]["model_dir"]
        )
        all_summaries[symbol_cfg["name"]] = summary
    return all_summaries
