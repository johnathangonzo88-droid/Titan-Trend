#!/usr/bin/env python3
"""
Backtesting script: run the strategy (optionally ML-augmented, using
leak-free out-of-sample probabilities from walk-forward validation) over
each symbol's full history and produce a performance report + charts under
reports/.

Usage:
    python scripts/backtest.py                  # all symbols
    python scripts/backtest.py --symbol ES       # one symbol
    python scripts/backtest.py --no-ml           # rules-only, no ML blend
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import BacktestEngine
from src.backtest.report import generate_backtest_report
from src.data_layer.loader import load_ohlcv_csv
from src.features.feature_engineering import build_features
from src.ml.dataset import build_ml_dataset
from src.ml.walkforward import generate_oos_probabilities, run_walk_forward_evaluation, select_best_model
from src.strategy.risk import position_size
from src.utils.config_loader import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def backtest_symbol(symbol_cfg: dict, config: dict, use_ml: bool, equity_override: float | None = None) -> dict:
    symbol = symbol_cfg["name"]
    logger.info("=== Backtesting %s ===", symbol)

    df, quality_report = load_ohlcv_csv(symbol_cfg["raw_file"], symbol)
    features = build_features(df, config["features"], has_volume=quality_report.has_volume)

    notes = list(quality_report.notes)
    ml_proba = None

    if use_ml and symbol_cfg.get("ml_enabled", False):
        X, y, _ = build_ml_dataset(features, config["ml"])
        results = run_walk_forward_evaluation(X, y, config["ml"])
        if results:
            best_name = select_best_model(results, metric="roc_auc")
            ml_proba = generate_oos_probabilities(X, y, best_name, config["ml"])
            notes.append(
                f"ML-augmented backtest using '{best_name}' with leak-free walk-forward "
                f"out-of-sample probabilities (mean ROC-AUC={results[best_name].mean_metrics['roc_auc']:.3f})."
            )
            features = features.loc[X.index]
        else:
            notes.append("No ML models available in this environment; ran rules-only.")
    else:
        notes.append("Rules-only backtest (ML not requested or not enabled for this symbol).")

    engine = BacktestEngine(
        features_df=features,
        strategy_cfg=config["strategy"],
        backtest_cfg=config["backtest"],
        point_value=symbol_cfg["point_value"],
        tick_size=symbol_cfg["tick_size"],
        ml_proba=ml_proba,
    )
    initial_equity = equity_override if equity_override is not None else config["backtest"]["initial_equity"]

    # Sanity check: is the account big enough to afford >=1 contract at this
    # symbol's typical ATR-based stop, given the configured risk-per-trade%?
    # This is a real constraint (bigger point-value instruments need bigger
    # accounts at the same risk %), surfaced here rather than silently
    # producing an empty trade log with no explanation.
    typical_atr = features["atr"].dropna().median()
    typical_risk_per_contract = (
        typical_atr * config["strategy"]["atr_stop_multiplier"] * symbol_cfg["point_value"]
    )
    affordable_contracts = position_size(
        initial_equity, config["strategy"]["risk_per_trade_pct"], typical_risk_per_contract
    )
    if affordable_contracts < 1:
        budget = initial_equity * config["strategy"]["risk_per_trade_pct"] / 100.0
        notes.append(
            f"Position sizing infeasible at ${initial_equity:,.0f} equity / "
            f"{config['strategy']['risk_per_trade_pct']}% risk: the typical stop-loss distance "
            f"(median ATR x {config['strategy']['atr_stop_multiplier']}) risks "
            f"~${typical_risk_per_contract:,.0f} per contract, above the ${budget:,.0f} risk budget - "
            f"so 0 (or very few) trades can be sized. Re-run with a larger --equity, a higher "
            f"risk_per_trade_pct in config.yaml, or trade this symbol's micro equivalent instead."
        )
    if equity_override is not None:
        notes.append(
            f"Backtested with ${equity_override:,.0f} starting equity (overriding config default of "
            f"${config['backtest']['initial_equity']:,.0f}) - see notes on position sizing feasibility."
        )
    trades, equity = engine.run(initial_equity=initial_equity)

    report_dir = resolve_path(config["backtest"]["report_dir"])
    metrics = generate_backtest_report(symbol, trades, equity, report_dir, extra_notes=notes)
    metrics["symbol"] = symbol
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the strategy on historical data.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--symbol", default=None, help="Only backtest this symbol (default: all)")
    parser.add_argument("--no-ml", action="store_true", help="Disable ML blending, rules-only")
    parser.add_argument("--equity", type=float, default=None, help="Override starting equity from config.yaml")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    symbols = config["symbols"]
    if args.symbol:
        symbols = [s for s in symbols if s["name"] == args.symbol]
        if not symbols:
            logger.error("Symbol '%s' not found in config", args.symbol)
            sys.exit(1)

    all_metrics = {}
    for symbol_cfg in symbols:
        all_metrics[symbol_cfg["name"]] = backtest_symbol(
            symbol_cfg, config, use_ml=not args.no_ml, equity_override=args.equity
        )

    print(json.dumps(all_metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
