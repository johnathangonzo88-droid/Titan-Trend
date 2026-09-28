"""Tests specifically targeting leak-free labeling and walk-forward
splitting, since this is the correctness property the whole ML brief hinges
on ("prevents lookahead bias", "prevents data leakage")."""
import numpy as np
import pandas as pd
import pytest

from src.ml.dataset import _triple_barrier_label
from src.ml.walkforward import generate_walk_forward_folds, time_series_cv_splits


def test_triple_barrier_label_hits_upper_first():
    # Price rises steadily -> should label 1 (upper barrier hit before lower)
    close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    high = close + 0.5
    low = close - 0.5
    atr = np.array([1.0] * len(close))
    labels = _triple_barrier_label(high, low, close, atr, horizon=3, atr_multiplier=1.0)
    assert labels[0] == 1.0


def test_triple_barrier_label_hits_lower_first():
    close = np.array([100.0, 99.0, 98.0, 97.0, 96.0, 95.0])
    high = close + 0.5
    low = close - 0.5
    atr = np.array([1.0] * len(close))
    labels = _triple_barrier_label(high, low, close, atr, horizon=3, atr_multiplier=1.0)
    assert labels[0] == 0.0


def test_triple_barrier_label_no_barrier_hit_is_zero():
    close = np.array([100.0] * 10)  # flat, never moves
    high = close + 0.1
    low = close - 0.1
    atr = np.array([1.0] * len(close))
    labels = _triple_barrier_label(high, low, close, atr, horizon=3, atr_multiplier=1.0)
    assert labels[0] == 0.0


def test_triple_barrier_label_last_rows_are_nan():
    n = 20
    close = np.linspace(100, 110, n)
    high, low = close + 0.5, close - 0.5
    atr = np.array([1.0] * n)
    horizon = 5
    labels = _triple_barrier_label(high, low, close, atr, horizon=horizon, atr_multiplier=1.0)
    assert np.all(np.isnan(labels[n - horizon :]))


def test_walk_forward_folds_never_overlap_train_and_test():
    folds = generate_walk_forward_folds(n_samples=1000, train_window=300, test_window=100, step=100, min_train=200)
    assert len(folds) > 0
    for fold in folds:
        assert fold.train_end <= fold.test_start  # no leakage: train strictly precedes test
        assert fold.test_start < fold.test_end


def test_walk_forward_folds_progress_forward_in_time():
    folds = generate_walk_forward_folds(n_samples=1000, train_window=300, test_window=100, step=100, min_train=200)
    for a, b in zip(folds, folds[1:]):
        assert b.train_start >= a.train_start  # each fold starts at or after the previous


def test_walk_forward_folds_empty_when_insufficient_data():
    folds = generate_walk_forward_folds(n_samples=50, train_window=300, test_window=100, step=100, min_train=200)
    assert folds == []


def test_time_series_cv_splits_train_precedes_test():
    for train_idx, test_idx in time_series_cv_splits(n_samples=600, n_splits=5):
        assert train_idx.max() < test_idx.min()  # strict time ordering, no leakage
