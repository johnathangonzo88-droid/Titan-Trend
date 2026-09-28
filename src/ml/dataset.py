"""
ML dataset construction: leak-free labeling.

This is the one place in the codebase allowed to look forward in time - and
only to build the *label*, never a feature. Everything in
src/features/feature_engineering.py is already strictly backward-looking
(see that module's docstring), so as long as this module keeps that
separation, the resulting dataset has no lookahead bias.

Label definition
-----------------
For each bar t, look `horizon_bars` forward and ask: did price move at least
`atr_multiplier * ATR[t]` in the favorable direction before moving that much
against it? This is a triple-barrier-style label (a simplified version of
the well-known "triple barrier method"):

    label = 1  if the forward high/low path hits +atr_multiplier*ATR first
    label = 0  if it hits -atr_multiplier*ATR first (or neither barrier is
               hit within the horizon - treated as a non-event / 0)

Using ATR-scaled barriers (rather than a fixed percentage) makes the label
comparable across volatility regimes and across instruments.

Train/serve consistency: the exact same feature columns used here are what
`src/ml/models.py` trains on and what `live_signal.py` must pass in at
inference time - enforced by always going through `feature_columns()`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.feature_engineering import feature_columns
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _triple_barrier_label(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray,
    horizon: int, atr_multiplier: float,
) -> np.ndarray:
    n = len(close)
    labels = np.full(n, np.nan)

    for t in range(n - horizon):
        if np.isnan(atr[t]) or atr[t] <= 0:
            continue
        entry = close[t]
        upper_barrier = entry + atr_multiplier * atr[t]
        lower_barrier = entry - atr_multiplier * atr[t]

        window_high = high[t + 1 : t + 1 + horizon]
        window_low = low[t + 1 : t + 1 + horizon]

        hit_upper = np.argmax(window_high >= upper_barrier) if np.any(window_high >= upper_barrier) else -1
        hit_lower = np.argmax(window_low <= lower_barrier) if np.any(window_low <= lower_barrier) else -1

        if hit_upper == -1 and hit_lower == -1:
            labels[t] = 0.0  # no significant move either way within horizon
        elif hit_upper == -1:
            labels[t] = 0.0  # only downside barrier hit
        elif hit_lower == -1:
            labels[t] = 1.0  # only upside barrier hit
        else:
            labels[t] = 1.0 if hit_upper < hit_lower else 0.0  # whichever hit first

    return labels


def build_ml_dataset(
    features_df: pd.DataFrame, ml_cfg: dict
) -> tuple[pd.DataFrame, pd.Series, list]:
    """Build (X, y, feature_cols) for one symbol's featured dataframe.

    Rows with any NaN feature (warm-up period) or an undefined label (the
    final `horizon_bars` rows, which have no forward window) are dropped.
    """
    cols = feature_columns(features_df)

    # One-hot encode the categorical regime label so tree models can split on it.
    regime_dummies = pd.get_dummies(features_df["regime"], prefix="regime")
    working = pd.concat([features_df[cols], regime_dummies], axis=1)
    cols_with_regime = cols + list(regime_dummies.columns)

    labels = _triple_barrier_label(
        high=features_df["high"].values,
        low=features_df["low"].values,
        close=features_df["close"].values,
        atr=features_df["atr"].values,
        horizon=ml_cfg["label_horizon_bars"],
        atr_multiplier=ml_cfg["label_atr_multiplier"],
    )
    working["label"] = labels

    before = len(working)
    working = working.dropna(subset=cols_with_regime + ["label"])
    logger.info(
        "ML dataset: %d/%d rows retained after dropping warm-up/undefined-label rows",
        len(working), before,
    )

    X = working[cols_with_regime].astype(float)
    y = working["label"].astype(int)
    logger.info("Label balance: %s", y.value_counts(normalize=True).to_dict())

    return X, y, cols_with_regime
