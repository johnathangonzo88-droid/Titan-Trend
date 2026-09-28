"""
Feature engineering orchestrator.

Combines every indicator in `indicators.py` plus the regime detector into a
single wide dataframe of features, aligned 1:1 with the input OHLCV bars.

Design principle for leak-free features: every column here is computed
using `.rolling()`, `.ewm()`, or `.shift()` over data at or before the
current bar. Nothing here looks forward. The ML dataset builder
(`src/ml/dataset.py`) is the ONLY place that intentionally looks forward,
and it does so exclusively to build the *label* (never a feature).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import indicators as ind
from src.features.regime import detect_regime
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_features(df: pd.DataFrame, feature_cfg: dict, has_volume: bool) -> pd.DataFrame:
    """Build the full feature matrix for one symbol.

    Parameters
    ----------
    df : canonical OHLCV dataframe (see src/data_layer/loader.py)
    feature_cfg : the `features:` section of config.yaml
    has_volume : whether `df['volume']` contains real trade volume (from the
                 DataQualityReport) - controls whether true VWAP/volume-zscore
                 or their documented proxies are computed.

    Returns
    -------
    DataFrame with the original OHLCV columns plus all engineered features.
    """
    out = df.copy()
    o, h, l, c = out["open"], out["high"], out["low"], out["close"]

    # --- Trend: EMAs / SMAs ---
    for p in feature_cfg["ema_periods"]:
        out[f"ema_{p}"] = ind.ema(c, p)
    for p in feature_cfg["sma_periods"]:
        out[f"sma_{p}"] = ind.sma(c, p)

    # Price position relative to key moving averages (normalized, scale-free)
    for p in feature_cfg["ema_periods"]:
        out[f"dist_ema_{p}_pct"] = (c - out[f"ema_{p}"]) / out[f"ema_{p}"] * 100.0

    # EMA slope (trend direction/strength over last 5 bars)
    fast_p = feature_cfg["ema_periods"][0]
    out[f"ema_{fast_p}_slope"] = out[f"ema_{fast_p}"].diff(5) / out[f"ema_{fast_p}"].shift(5) * 100.0

    # --- Momentum ---
    out["rsi"] = ind.rsi(c, feature_cfg["rsi_period"])
    stoch_k, stoch_d = ind.stochastic_rsi(
        c, feature_cfg["stoch_rsi_period"], feature_cfg["stoch_k"], feature_cfg["stoch_d"]
    )
    out["stoch_rsi_k"] = stoch_k
    out["stoch_rsi_d"] = stoch_d

    macd_line, macd_signal, macd_hist = ind.macd(
        c, feature_cfg["macd_fast"], feature_cfg["macd_slow"], feature_cfg["macd_signal"]
    )
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist
    out["macd_hist_rising"] = (macd_hist.diff() > 0).astype(float)

    # --- Volatility ---
    out["atr"] = ind.atr(h, l, c, feature_cfg["atr_period"])
    out["atr_pct"] = out["atr"] / c * 100.0

    bb_mid, bb_upper, bb_lower = ind.bollinger_bands(
        c, feature_cfg["bollinger_period"], feature_cfg["bollinger_std"]
    )
    out["bb_mid"], out["bb_upper"], out["bb_lower"] = bb_mid, bb_upper, bb_lower
    out["bb_bandwidth"] = ind.bollinger_bandwidth(bb_upper, bb_lower, bb_mid)
    out["bb_pct_b"] = (c - bb_lower) / (bb_upper - bb_lower).replace(0.0, np.nan)

    # --- Trend strength ---
    adx_val, plus_di, minus_di = ind.adx(h, l, c, feature_cfg["adx_period"])
    out["adx"], out["plus_di"], out["minus_di"] = adx_val, plus_di, minus_di
    out["di_diff"] = plus_di - minus_di

    # --- Volume / participation (real or proxy, always labeled) ---
    out["has_volume"] = has_volume
    if has_volume:
        out["vwap"] = ind.true_vwap(h, l, c, out["volume"], window=feature_cfg["vwap_window"])
        out["volume_z"] = ind.volume_zscore(out["volume"], feature_cfg["volume_proxy_window"])
    else:
        out["vwap_proxy"] = ind.proxy_vwap(h, l, c, window=feature_cfg["vwap_window"])
        out["participation_proxy_z"] = ind.participation_proxy(
            h, l, c, o, window=feature_cfg["volume_proxy_window"]
        )

    dist_to_vwap_source = out["vwap"] if has_volume else out["vwap_proxy"]
    out["dist_vwap_pct"] = (c - dist_to_vwap_source) / dist_to_vwap_source * 100.0

    # --- Price action ---
    out["bar_range"] = h - l
    out["body"] = (c - o).abs()
    out["body_to_range"] = out["body"] / out["bar_range"].replace(0.0, np.nan)
    out["upper_wick"] = h - out[["open", "close"]].max(axis=1)
    out["lower_wick"] = out[["open", "close"]].min(axis=1) - l
    out["is_bullish"] = (c > o).astype(float)
    out["return_1"] = c.pct_change(1) * 100.0
    out["return_5"] = c.pct_change(5) * 100.0

    # --- Regime ---
    out["regime"] = detect_regime(
        out,
        adx_period=feature_cfg["adx_period"],
        atr_period=feature_cfg["atr_period"],
        lookback=feature_cfg["regime_lookback"],
    )

    n_total = len(out)
    n_warmup = out["adx"].isna().sum()
    logger.info(
        "Built %d features; %d/%d rows are warm-up NaNs and will be excluded downstream",
        out.shape[1] - df.shape[1], n_warmup, n_total,
    )

    return out


def feature_columns(df: pd.DataFrame) -> list:
    """Return the list of numeric feature columns suitable for ML input
    (excludes raw OHLCV, volume/has_volume flags, and the categorical regime
    label, which is one-hot encoded separately by the ML dataset builder)."""
    exclude = {"open", "high", "low", "close", "volume", "has_volume", "regime"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
