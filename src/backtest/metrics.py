"""
Performance metrics for a completed backtest trade log.

Every function takes a trade-level DataFrame (one row per closed trade, with
at least a `pnl` column in dollars and `bars_held`) and/or an equity curve
Series, and returns a plain float - kept separate from the backtest engine
so metrics can be unit-tested against known values independent of any
simulation logic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def win_rate(trades: pd.DataFrame) -> float:
    if len(trades) == 0 or "pnl" not in trades.columns:
        return 0.0
    return float((trades["pnl"] > 0).mean())


def profit_factor(trades: pd.DataFrame) -> float:
    if len(trades) == 0 or "pnl" not in trades.columns:
        return 0.0
    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def average_trade_return(trades: pd.DataFrame) -> float:
    if len(trades) == 0 or "pnl" not in trades.columns:
        return 0.0
    return float(trades["pnl"].mean())


def total_return(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)


def max_drawdown(equity_curve: pd.Series) -> tuple[float, float]:
    """Returns (max_drawdown_pct, max_drawdown_dollars) - both negative or zero."""
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    dd_dollars = equity_curve - running_max
    return float(drawdown.min()), float(dd_dollars.min())


def _periods_per_year(equity_curve: pd.Series) -> float:
    """Infer annualization factor from the equity curve's own time index."""
    if len(equity_curve) < 2:
        return 252.0
    total_seconds = (equity_curve.index[-1] - equity_curve.index[0]).total_seconds()
    if total_seconds <= 0:
        return 252.0
    avg_step_seconds = total_seconds / (len(equity_curve) - 1)
    seconds_per_year = 365.25 * 24 * 3600
    return seconds_per_year / avg_step_seconds


def sharpe_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio computed from equity-curve period returns."""
    returns = equity_curve.pct_change().dropna()
    if len(returns) < 2 or returns.std(ddof=0) == 0:
        return 0.0
    periods_per_year = _periods_per_year(equity_curve)
    excess = returns - (risk_free_rate / periods_per_year)
    return float(np.sqrt(periods_per_year) * excess.mean() / returns.std(ddof=0))


def sortino_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sortino ratio - like Sharpe but only penalizes downside
    volatility, which better matches "risk-adjusted returns" for strategies
    with deliberately asymmetric win/loss distributions."""
    returns = equity_curve.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    periods_per_year = _periods_per_year(equity_curve)
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=0)
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    excess = returns - (risk_free_rate / periods_per_year)
    return float(np.sqrt(periods_per_year) * excess.mean() / downside_std)


def summarize(trades: pd.DataFrame, equity_curve: pd.Series) -> dict:
    """Compute every metric required by the brief in one call."""
    dd_pct, dd_dollars = max_drawdown(equity_curve)
    return {
        "n_trades": int(len(trades)),
        "win_rate": win_rate(trades),
        "sharpe_ratio": sharpe_ratio(equity_curve),
        "sortino_ratio": sortino_ratio(equity_curve),
        "max_drawdown_pct": dd_pct,
        "max_drawdown_dollars": dd_dollars,
        "profit_factor": profit_factor(trades),
        "total_return_pct": total_return(equity_curve),
        "avg_trade_return_dollars": average_trade_return(trades),
        "final_equity": float(equity_curve.iloc[-1]) if len(equity_curve) else 0.0,
    }
