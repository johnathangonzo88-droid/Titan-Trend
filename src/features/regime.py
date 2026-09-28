"""
Market regime detection.

Classifies each bar into one of four regimes using only backward-looking
data (no lookahead): trending, ranging, volatile, low_volatility. The
strategy module uses this to decide which setups are allowed to fire (e.g.
trend-following entries are suppressed in a "ranging" regime).

Classification logic (applied in this priority order per bar):
  1. volatile         : ATR% (ATR / close) is in its own top decile recently
                         AND Bollinger bandwidth is expanding fast.
  2. low_volatility    : ATR% is in its own bottom decile recently (a "squeeze").
  3. trending          : ADX above a threshold (default 25).
  4. ranging           : everything else (ADX low, volatility normal).

Thresholds are computed from each series' own rolling history (percentile
based) rather than hard-coded absolute levels, so the same logic is valid
across instruments with very different price scales (ES vs NQ, etc.).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import indicators as ind


REGIME_TRENDING = "trending"
REGIME_RANGING = "ranging"
REGIME_VOLATILE = "volatile"
REGIME_LOW_VOL = "low_volatility"


def detect_regime(
    df: pd.DataFrame,
    adx_period: int = 14,
    atr_period: int = 14,
    lookback: int = 50,
    adx_trend_threshold: float = 25.0,
    high_vol_percentile: float = 0.85,
    low_vol_percentile: float = 0.15,
) -> pd.Series:
    """Return a categorical Series of regime labels aligned to df.index.

    `df` must already contain the OHLC columns; this function computes its
    own ADX/ATR internally (kept independent of the main feature builder so
    it can be unit-tested and reused standalone).
    """
    high, low, close = df["high"], df["low"], df["close"]

    adx_val, _, _ = ind.adx(high, low, close, period=adx_period)
    atr_val = ind.atr(high, low, close, period=atr_period)
    atr_pct = atr_val / close.replace(0.0, np.nan) * 100.0

    # Rolling percentile rank of current ATR% vs. its own recent history.
    atr_pct_rank = atr_pct.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    _, upper, lower = ind.bollinger_bands(close, period=20, num_std=2.0)
    mid = ind.sma(close, 20)
    bandwidth = ind.bollinger_bandwidth(upper, lower, mid)
    bandwidth_delta = bandwidth.diff(5)  # expansion speed over 5 bars

    regime = pd.Series(index=df.index, dtype="object")

    is_volatile = (atr_pct_rank >= high_vol_percentile) & (bandwidth_delta > 0)
    is_low_vol = atr_pct_rank <= low_vol_percentile
    is_trending = adx_val >= adx_trend_threshold

    regime[:] = REGIME_RANGING
    regime[is_trending] = REGIME_TRENDING
    regime[is_low_vol & ~is_volatile] = REGIME_LOW_VOL
    regime[is_volatile] = REGIME_VOLATILE  # highest priority, applied last

    # Rows where inputs aren't warmed up yet stay NaN rather than a guessed label
    warmup_mask = adx_val.isna() | atr_pct_rank.isna()
    regime[warmup_mask] = np.nan

    return regime
