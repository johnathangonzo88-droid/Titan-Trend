"""
Walk-forward validation.

Standard k-fold cross-validation shuffles data and is invalid for time
series: it lets a model trained partly on the future predict the past. This
module instead implements walk-forward validation - a sequence of expanding
or rolling train windows, each followed by an out-of-sample test window that
comes strictly *after* it in time, which is what the brief requires
("prevents lookahead bias", "uses walk-forward validation").

    |--- train fold 1 ---|-- test 1 --|
              |--- train fold 2 (rolled forward) ---|-- test 2 --|
                        |--- train fold 3 ---|-- test 3 --|
                                            ...

Additionally, `time_series_cv_splits` provides a plain expanding-window
time-series CV generator (scikit-learn's `TimeSeriesSplit` semantics) used
for hyperparameter selection within a single walk-forward training window,
satisfying the "time-series cross validation" requirement without mixing
concerns with the outer walk-forward evaluation loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.ml.models import ModelResult, get_available_models, train_and_evaluate_fold
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WalkForwardFold:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def generate_walk_forward_folds(
    n_samples: int, train_window: int, test_window: int, step: int, min_train: int
) -> list:
    """Rolling-window walk-forward split indices (integer positions, not
    timestamps, so this works regardless of gaps/session breaks in the index).
    """
    folds = []
    train_start = 0
    while True:
        train_end = train_start + max(train_window, min_train)
        test_start = train_end
        test_end = test_start + test_window
        if test_end > n_samples:
            break
        folds.append(WalkForwardFold(train_start, train_end, test_start, test_end))
        train_start += step
    return folds


def time_series_cv_splits(n_samples: int, n_splits: int = 5):
    """Expanding-window time series CV splits (sklearn TimeSeriesSplit
    semantics), for use *within* one walk-forward training window - e.g. for
    model/hyperparameter selection without touching the held-out test fold.
    """
    fold_size = n_samples // (n_splits + 1)
    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_end = fold_size * (i + 1)
        yield np.arange(0, train_end), np.arange(train_end, test_end)


def run_walk_forward_evaluation(
    X: pd.DataFrame, y: pd.Series, ml_cfg: dict
) -> dict:
    """Run every requested-and-available model through the same walk-forward
    folds and return {model_name: ModelResult}.

    Using identical folds for every model is what makes the model comparison
    fair - each candidate sees exactly the same train/test splits.
    """
    wf_cfg = ml_cfg["walk_forward"]
    folds = generate_walk_forward_folds(
        n_samples=len(X),
        train_window=wf_cfg["train_window_bars"],
        test_window=wf_cfg["test_window_bars"],
        step=wf_cfg["step_bars"],
        min_train=wf_cfg["min_train_bars"],
    )
    if not folds:
        raise ValueError(
            f"Not enough data ({len(X)} rows) to form even one walk-forward fold "
            f"with train_window={wf_cfg['train_window_bars']}, test_window={wf_cfg['test_window_bars']}. "
            f"Reduce these in config.yaml or provide more history."
        )
    logger.info("Generated %d walk-forward folds", len(folds))

    factories = get_available_models(ml_cfg["models"])
    logger.info("Models available in this environment: %s", list(factories.keys()))

    results: dict = {}
    X_values = X.values
    y_values = y.values

    for name, factory in factories.items():
        fold_metrics = []
        last_model = None
        for i, fold in enumerate(folds):
            X_train = X_values[fold.train_start : fold.train_end]
            y_train = y_values[fold.train_start : fold.train_end]
            X_test = X_values[fold.test_start : fold.test_end]
            y_test = y_values[fold.test_start : fold.test_end]

            if len(np.unique(y_train)) < 2:
                logger.warning("Fold %d for %s has a single class in train; skipping fold", i, name)
                continue

            metrics, model = train_and_evaluate_fold(factory, X_train, y_train, X_test, y_test)
            metrics["fold"] = i
            metrics["n_train"] = len(y_train)
            metrics["n_test"] = len(y_test)
            fold_metrics.append(metrics)
            last_model = model

        if not fold_metrics:
            logger.warning("No valid folds for model %s; skipping", name)
            continue

        df_metrics = pd.DataFrame(fold_metrics)
        mean_metrics = df_metrics[["accuracy", "precision", "recall", "f1", "roc_auc"]].mean().to_dict()
        logger.info("%s: mean walk-forward metrics = %s", name, {k: round(v, 4) for k, v in mean_metrics.items()})

        results[name] = ModelResult(
            name=name, fold_metrics=fold_metrics, mean_metrics=mean_metrics, model=last_model
        )

    return results


def generate_oos_probabilities(
    X: pd.DataFrame, y: pd.Series, model_name: str, ml_cfg: dict
) -> pd.Series:
    """Produce a leak-free, out-of-sample probability series for backtesting.

    For every walk-forward fold, fits `model_name` on that fold's train
    window only and scores that fold's test window. Stitches the per-fold
    test predictions into one Series aligned to X's index. Bars before the
    first fold's test window (i.e. the initial training period, which never
    gets an out-of-sample prediction) are left as NaN - the backtest engine
    treats NaN as "no ML input available", falling back to rules only,
    which is the honest behavior rather than fabricating a probability.
    """
    wf_cfg = ml_cfg["walk_forward"]
    folds = generate_walk_forward_folds(
        n_samples=len(X),
        train_window=wf_cfg["train_window_bars"],
        test_window=wf_cfg["test_window_bars"],
        step=wf_cfg["step_bars"],
        min_train=wf_cfg["min_train_bars"],
    )
    factories = get_available_models([model_name])
    if model_name not in factories:
        raise ValueError(f"Model '{model_name}' is not available in this environment.")
    factory = factories[model_name]

    proba = pd.Series(np.nan, index=X.index)
    X_values = X.values
    y_values = y.values

    for fold in folds:
        X_train = X_values[fold.train_start : fold.train_end]
        y_train = y_values[fold.train_start : fold.train_end]
        X_test = X_values[fold.test_start : fold.test_end]

        if len(np.unique(y_train)) < 2:
            continue

        model = factory()
        model.fit(X_train, y_train)
        test_proba = model.predict_proba(X_test)[:, 1]
        proba.iloc[fold.test_start : fold.test_end] = test_proba

    n_covered = proba.notna().sum()
    logger.info(
        "Generated out-of-sample ML probabilities for %d/%d bars (%s uncovered = rules-only fallback)",
        n_covered, len(proba), len(proba) - n_covered,
    )
    return proba


def select_best_model(results: dict, metric: str = "roc_auc") -> str:
    """Pick the model with the highest mean out-of-sample `metric`.

    ROC-AUC (rather than accuracy) is the default selection metric because
    the label distribution can be imbalanced and AUC is threshold-independent
    - consistent with optimizing for risk-adjusted quality over raw hit rate.
    """
    if not results:
        raise ValueError("No model results to select from.")
    best_name = max(results, key=lambda n: results[n].mean_metrics.get(metric, float("-inf")))
    logger.info("Selected best model: %s (%s=%.4f)", best_name, metric, results[best_name].mean_metrics[metric])
    return best_name
