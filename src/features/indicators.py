"""
Technical indicators, implemented from scratch with pandas/numpy only.

Rationale for a hand-rolled implementation rather than a third-party TA
library: it keeps the dependency footprint minimal (works in restricted
network environments), and every formula is visible and testable in this
one file. Each function:
  * takes a canonical OHLCV dataframe (see src/data_layer/loader.py),
  * is a pure function (no hidden state, no lookahead),
  * uses only data at or before index i to compute row i, which is what
    prevents lookahead bias downstream in the ML pipeline.

All rolling windows use `min_periods=window` (or the natural minimum for the
formula) so early rows are NaN rather than silently using a partial window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (adjust=False matches standard trading-platform EMA)."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing = EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # Where avg_loss is exactly 0 (all gains), RSI = 100
    out = out.where(avg_loss != 0.0, 100.0)
    return out


def stochastic_rsi(
    close: pd.Series, rsi_period: int = 14, k_period: int = 3, d_period: int = 3
) -> tuple[pd.Series, pd.Series]:
    """Stochastic RSI (%K, %D), oscillating 0-100.

    Stochastic applied to RSI values rather than price - more sensitive to
    momentum shifts than plain RSI, at the cost of more noise.
    """
    rsi_vals = rsi(close, rsi_period)
    lowest = rsi_vals.rolling(window=rsi_period, min_periods=rsi_period).min()
    highest = rsi_vals.rolling(window=rsi_period, min_periods=rsi_period).max()
    stoch = 100.0 * (rsi_vals - lowest) / (highest - lowest).replace(0.0, np.nan)
    k = stoch.rolling(window=k_period, min_periods=k_period).mean()
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k, d


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram."""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder's smoothing)."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def bollinger_bands(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: middle (SMA), upper, lower."""
    mid = sma(close, period)
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def bollinger_bandwidth(upper: pd.Series, lower: pd.Series, mid: pd.Series) -> pd.Series:
    """Normalized band width - a standard volatility/squeeze proxy."""
    return (upper - lower) / mid.replace(0.0, np.nan)


# ---------------------------------------------------------------------------
# Trend strength
# ---------------------------------------------------------------------------

def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Index, plus +DI/-DI.

    Returns (adx, plus_di, minus_di). ADX > 25 is conventionally "trending",
    below 20 "ranging" - used by the regime detector.
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr = true_range(high, low, close)
    atr_smoothed = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * (
        plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        / atr_smoothed.replace(0.0, np.nan)
    )
    minus_di = 100.0 * (
        minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        / atr_smoothed.replace(0.0, np.nan)
    )

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_val = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    return adx_val, plus_di, minus_di


# ---------------------------------------------------------------------------
# Volume / participation
# ---------------------------------------------------------------------------
# IMPORTANT: none of the four provided data files contain genuine trade
# volume (see src/data_layer/loader.py DataQualityReport). The functions
# below implement:
#   1) a *real* VWAP/volume path used automatically when a `volume` column
#      with real data is present, and
#   2) a *proxy* path (participation estimated from bar range and directional
#      strength) used automatically when it is not.
# Every caller uses `has_volume` to know which path was used - there is no
# silent substitution that could be mistaken for real volume confirmation.

def true_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
              window: int | None = None) -> pd.Series:
    """Volume-weighted average price using real volume.

    If `window` is None, computes a session-cumulative VWAP (resets are the
    caller's responsibility, e.g. group by trading session). If `window` is
    given, computes a rolling VWAP instead, which is what this project uses
    (simpler and avoids needing explicit session boundaries).
    """
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    if window is None:
        return pv.cumsum() / volume.cumsum().replace(0.0, np.nan)
    return (
        pv.rolling(window=window, min_periods=window).sum()
        / volume.rolling(window=window, min_periods=window).sum().replace(0.0, np.nan)
    )


def proxy_vwap(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling typical-price average used as a VWAP substitute when no real
    volume is available. This is NOT volume-weighted - it is documented and
    named as a proxy so it is never confused with a true VWAP in signal
    reasoning or reports.
    """
    typical_price = (high + low + close) / 3.0
    return typical_price.rolling(window=window, min_periods=window).mean()


def volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    """Rolling z-score of real volume - used to detect genuine volume spikes."""
    mean = volume.rolling(window=window, min_periods=window).mean()
    std = volume.rolling(window=window, min_periods=window).std(ddof=0)
    return (volume - mean) / std.replace(0.0, np.nan)


def participation_proxy(
    high: pd.Series, low: pd.Series, close: pd.Series, open_: pd.Series, window: int = 20
) -> pd.Series:
    """A z-scored 'conviction' proxy used in place of a volume spike when no
    real volume exists. Combines:
      * bar range relative to its recent average (expansion = more participants), and
      * body-to-range ratio (a strong close near the extreme = directional conviction).

    This is explicitly a proxy for order-flow intensity, not a measurement of
    actual traded volume, and every place it's used in the strategy/report
    labels it "volume-proxy" rather than "volume".
    """
    bar_range = (high - low).replace(0.0, np.nan)
    body = (close - open_).abs()
    body_ratio = body / bar_range
    range_z = volume_zscore(bar_range, window)  # reuse the same z-score machinery
    conviction = range_z.fillna(0.0) * 0.6 + (body_ratio.fillna(0.0) - 0.5) * 2 * 0.4
    return conviction
