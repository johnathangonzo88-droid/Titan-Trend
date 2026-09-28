"""Unit tests for src/backtest/metrics.py against hand-computed values."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import (
    average_trade_return,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    summarize,
    total_return,
    win_rate,
)


def _trades(pnls):
    return pd.DataFrame({"pnl": pnls})


def test_win_rate_simple():
    trades = _trades([100, -50, 100, -50, 100])
    assert win_rate(trades) == 0.6


def test_win_rate_empty():
    assert win_rate(_trades([])) == 0.0


def test_profit_factor_simple():
    trades = _trades([100, 100, -50])
    assert profit_factor(trades) == pytest.approx(200 / 50)


def test_profit_factor_no_losses_is_inf():
    trades = _trades([100, 50])
    assert profit_factor(trades) == float("inf")


def test_profit_factor_no_trades_is_zero():
    assert profit_factor(_trades([])) == 0.0


def test_average_trade_return():
    trades = _trades([100, -50, 150])
    assert average_trade_return(trades) == pytest.approx(66.6667, rel=1e-3)


def test_total_return():
    equity = pd.Series([100.0, 110.0, 121.0])
    assert total_return(equity) == pytest.approx(0.21)


def test_max_drawdown_detects_peak_to_trough():
    equity = pd.Series([100.0, 120.0, 90.0, 95.0, 130.0])
    dd_pct, dd_dollars = max_drawdown(equity)
    assert dd_pct == pytest.approx((90.0 - 120.0) / 120.0)
    assert dd_dollars == pytest.approx(90.0 - 120.0)


def test_sharpe_ratio_zero_for_flat_equity():
    idx = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    equity = pd.Series([100.0] * 10, index=idx)
    assert sharpe_ratio(equity) == 0.0


def test_sharpe_ratio_positive_for_steady_gains():
    idx = pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC")
    equity = pd.Series(np.linspace(100, 130, 30), index=idx)
    assert sharpe_ratio(equity) > 0


def test_sortino_ratio_ignores_upside_volatility():
    idx = pd.date_range("2026-01-01", periods=12, freq="D", tz="UTC")
    # Big, variably-sized up moves plus small, variably-sized down moves:
    # Sharpe is dragged down by the large upside swings (they count as
    # "volatility"), Sortino should not be, since downside deviation only
    # sees the (varied, non-degenerate) small losses.
    multipliers = [1.06, 0.99, 1.04, 0.985, 1.07, 0.992, 1.05, 0.988, 1.03, 0.995, 1.045]
    values = [100.0]
    for m in multipliers:
        values.append(values[-1] * m)
    equity = pd.Series(values, index=idx)
    assert sortino_ratio(equity) > sharpe_ratio(equity)


def test_summarize_returns_all_required_keys():
    idx = pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC")
    trades = _trades([100, -50, 200])
    equity = pd.Series([50000, 50100, 50050, 50250], index=idx[:4])
    metrics = summarize(trades, equity)
    required = {
        "n_trades", "win_rate", "sharpe_ratio", "sortino_ratio", "max_drawdown_pct",
        "max_drawdown_dollars", "profit_factor", "total_return_pct",
        "avg_trade_return_dollars", "final_equity",
    }
    assert required.issubset(metrics.keys())


def test_summarize_handles_zero_trades():
    idx = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    equity = pd.Series([50000, 50000, 50000], index=idx)
    metrics = summarize(pd.DataFrame(), equity)
    assert metrics["n_trades"] == 0
    assert metrics["win_rate"] == 0.0
    assert metrics["profit_factor"] == 0.0
