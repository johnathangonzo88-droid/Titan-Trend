"""
Signal bot orchestrator: ties together data, features, strategy, ML, and the
Telegram client into the "check for a new signal, broadcast it" workflow.

Two ways to run this (see scripts/live_signal.py):
  1. One-shot (`run_once`) - ideal for a GitHub Actions / cron schedule that
     invokes the script every N minutes and exits (matches the polling
     pattern this project already uses in production).
  2. Continuous (`run_forever`) - a long-lived process that sleeps
     `poll_interval_seconds` between checks, for a Docker/VPS deployment.

Live data source: this reference implementation reads the same historical
CSVs used for backtesting and evaluates the strategy on their most recent
bar(s), which lets you validate the whole pipeline (features -> strategy ->
ML -> Telegram message) end-to-end without a live feed. Swap in a real
broker/data-vendor feed by implementing `LiveDataSource` in
`src/data_layer/live_feed.py` (stubbed) and passing it to `SignalBot`
instead of `CsvReplayDataSource`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from src.data_layer.loader import load_ohlcv_csv
from src.features.feature_engineering import build_features
from src.strategy.strategy import BUY, HOLD, SELL, generate_signal
from src.strategy.risk import position_size
from src.telegram_bot.client import TelegramClient
from src.telegram_bot.formatter import format_signal_message
from src.utils.config_loader import resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CsvReplayDataSource:
    """Feeds the bot from the same historical CSV files used for training/
    backtesting. Each call to `latest_bars()` re-reads the file and returns
    it in full; for a real deployment, replace this with a class that pulls
    only new bars from a live feed since the last check (see
    `src/data_layer/live_feed.py`).
    """

    def __init__(self, raw_file: str):
        self.raw_file = raw_file

    def latest_bars(self, symbol: str) -> tuple[pd.DataFrame, bool]:
        df, report = load_ohlcv_csv(self.raw_file, symbol)
        return df, report.has_volume


class SignalBot:
    def __init__(self, config: dict, settings, data_source_factory=None):
        self.config = config
        self.settings = settings
        self.telegram = TelegramClient(
            bot_token=settings.telegram_bot_token,
            parse_mode=config["telegram"]["parse_mode"],
        )
        self.data_source_factory = data_source_factory or (
            lambda symbol_cfg: CsvReplayDataSource(symbol_cfg["raw_file"])
        )
        self._models: dict = {}
        self._last_signaled_timestamp: dict = {}
        self._load_models()

    def _load_models(self) -> None:
        model_dir = resolve_path(self.config["ml"]["model_dir"])
        for symbol_cfg in self.config["symbols"]:
            if not symbol_cfg.get("ml_enabled", False):
                continue
            symbol = symbol_cfg["name"]
            matches = list(Path(model_dir).glob(f"{symbol}_*.joblib"))
            if not matches:
                logger.warning("No trained model found for %s in %s; will run rules-only for this symbol", symbol, model_dir)
                continue
            bundle = joblib.load(matches[0])
            self._models[symbol] = bundle
            logger.info("Loaded model for %s: %s (%s)", symbol, matches[0].name, bundle["model_name"])

    def _ml_probability(self, symbol: str, feature_row: pd.Series) -> Optional[float]:
        bundle = self._models.get(symbol)
        if bundle is None:
            return None
        cols = bundle["feature_cols"]
        # Build the same one-hot regime columns used at training time; any
        # regime category not seen at inference gets 0 for all dummies,
        # exactly matching pd.get_dummies' training-time behavior for an
        # unseen row (rather than crashing on a KeyError).
        row = {}
        for c in cols:
            if c.startswith("regime_"):
                row[c] = 1.0 if c == f"regime_{feature_row.get('regime')}" else 0.0
            else:
                row[c] = feature_row.get(c, 0.0)
        X = pd.DataFrame([row])[cols].astype(float)
        proba = bundle["model"].predict_proba(X.values)[:, 1]
        return float(proba[0])

    def evaluate_symbol(self, symbol_cfg: dict):
        """Run the full pipeline for one symbol's latest bar and return
        (SignalResult, feature_row, is_new_bar)."""
        symbol = symbol_cfg["name"]
        source = self.data_source_factory(symbol_cfg)
        df, has_volume = source.latest_bars(symbol)

        features = build_features(df, self.config["features"], has_volume=has_volume)
        latest = features.iloc[-1]
        latest_ts = features.index[-1]

        is_new_bar = self._last_signaled_timestamp.get(symbol) != latest_ts
        self._last_signaled_timestamp[symbol] = latest_ts

        if pd.isna(latest.get("adx")):
            logger.info("%s: latest bar still in warm-up period, skipping", symbol)
            return None, latest, is_new_bar

        ml_proba = self._ml_probability(symbol, latest)
        signal = generate_signal(
            row=latest,
            cfg=self.config["strategy"],
            point_value=symbol_cfg["point_value"],
            ml_probability=ml_proba,
        )
        return signal, latest, is_new_bar

    def run_once(self, account_equity: Optional[float] = None) -> list:
        """Evaluate every configured symbol once and broadcast any new
        BUY/SELL signal. Returns the list of messages sent (for testing/logging).
        """
        equity = account_equity or self.config["backtest"]["initial_equity"]
        sent = []

        for symbol_cfg in self.config["symbols"]:
            symbol = symbol_cfg["name"]
            try:
                signal, row, is_new_bar = self.evaluate_symbol(symbol_cfg)
            except Exception:
                logger.exception("Failed to evaluate %s; skipping this cycle", symbol)
                continue

            if signal is None or signal.action == HOLD:
                continue
            if not is_new_bar:
                logger.info("%s: signal unchanged from last check (same bar), not re-sending", symbol)
                continue

            contracts = position_size(
                account_equity=equity,
                risk_per_trade_pct=self.config["strategy"]["risk_per_trade_pct"],
                risk_per_contract=signal.risk_levels.risk_per_contract,
            )

            message = format_signal_message(
                symbol=symbol,
                timeframe=symbol_cfg["timeframe"],
                action=signal.action,
                entry=signal.risk_levels.entry,
                stop_loss=signal.risk_levels.stop_loss,
                take_profit_1=signal.risk_levels.take_profit_1,
                take_profit_2=signal.risk_levels.take_profit_2,
                reward_to_risk_tp2=signal.risk_levels.reward_to_risk_tp2,
                rule_score=signal.rule_score,
                reasons=signal.reasons,
                tick_size=symbol_cfg["tick_size"],
                contracts=contracts,
                strategy_name=self.config["project"].get("strategy_name"),
                session_date=row.name.strftime("%Y-%m-%d") if hasattr(row.name, "strftime") else str(row.name),
            )
            logger.info("Broadcasting %s signal for %s:\n%s", signal.action, symbol, message)
            self.telegram.broadcast(self.settings.telegram_chat_ids, message)
            sent.append(message)

        return sent

    def run_forever(self) -> None:
        interval = self.config["telegram"]["poll_interval_seconds"]
        logger.info("Starting continuous signal loop (interval=%ds)", interval)
        while True:
            self.run_once()
            time.sleep(interval)
