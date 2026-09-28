"""
Model zoo and a common training/evaluation interface.

Evaluates XGBoost, LightGBM, CatBoost, Random Forest, and Gradient Boosting
as specified in the brief. XGBoost/LightGBM/CatBoost are optional
dependencies: if a package isn't installed in the current environment, that
model is skipped with a clear log message rather than crashing the run, so
this file works unmodified whether or not the person running it has those
libraries installed. (In the sandbox this project was built in, outbound
package installs were blocked, so only the sklearn-native models actually
trained there - see reports/model_comparison.md for exactly what ran.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

# --- Optional heavy dependencies -------------------------------------------
try:
    from xgboost import XGBClassifier

    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier

    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier

    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False


@dataclass
class ModelResult:
    name: str
    fold_metrics: list          # list of dicts, one per walk-forward fold
    mean_metrics: dict
    model: object                # the last-fold-fitted estimator (for feature importance)


def get_available_models(requested: list) -> dict:
    """Instantiate the subset of `requested` model names that are actually
    installed in this environment. Always includes at least the sklearn
    models, which ship with scikit-learn and need no extra install.
    """
    factories = {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.05, max_iter=300, random_state=42,
        ),
    }
    if _HAS_XGBOOST:
        factories["xgboost"] = lambda: XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
    if _HAS_LIGHTGBM:
        factories["lightgbm"] = lambda: LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=-1,
        )
    if _HAS_CATBOOST:
        factories["catboost"] = lambda: CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05, random_state=42, verbose=False,
        )

    available = {}
    for name in requested:
        if name in factories:
            available[name] = factories[name]
        else:
            logger.warning(
                "Model '%s' requested in config but its package isn't installed; skipping. "
                "Install it (pip install %s) to include it.", name, name.replace("_", "-"),
            )
    return available


def _evaluate(y_true, y_pred, y_proba) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    # ROC-AUC needs both classes present in y_true
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def train_and_evaluate_fold(model_factory, X_train, y_train, X_test, y_test) -> tuple[dict, object]:
    """Fit one model on one walk-forward fold and return its test metrics."""
    model = model_factory()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = _evaluate(y_test, y_pred, y_proba)
    return metrics, model
