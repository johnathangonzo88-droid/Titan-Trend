"""Unit tests for src/features/indicators.py.

These check known/derivable properties (e.g. RSI is bounded 0-100, SMA of a
constant series equals that constant) rather than hard-coded magic numbers
scraped from another platform, since indicator formulas can legitimately
differ slightly in smoothing convention between platforms.
"""
import numpy as np
import pandas as pd
import pytest

from src.features import indicators as ind


@pytest.fixture
def sample_ohlc():
    n = 200
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.uniform(100, 1000, n)
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


def test_sma_constant_series():
    s = pd.Series([5.0] * 30)
    result = ind.sma(s, 10)
    assert np.isclose(result.iloc[-1], 5.0)
    assert result.iloc[:9].isna().all()  # warm-up period is NaN


def test_ema_reacts_faster_than_sma_to_a_jump():
    s = pd.Series([10.0] * 30 + [20.0] * 30)
    sma_val = ind.sma(s, 10).iloc[35]
    ema_val = ind.ema(s, 10).iloc[35]
    assert ema_val > sma_val  # EMA weights recent (higher) values more


def test_rsi_bounded(sample_ohlc):
    rsi = ind.rsi(sample_ohlc["close"], period=14)
    valid = rsi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_all_gains_is_100():
    s = pd.Series(np.arange(1, 30, dtype=float))  # strictly increasing
    rsi = ind.rsi(s, period=14)
    assert np.isclose(rsi.iloc[-1], 100.0)


def test_macd_histogram_is_macd_minus_signal(sample_ohlc):
    macd_line, signal, hist = ind.macd(sample_ohlc["close"])
    diff = (macd_line - signal - hist).dropna()
    assert np.allclose(diff, 0.0, atol=1e-9)


def test_atr_non_negative(sample_ohlc):
    atr = ind.atr(sample_ohlc["high"], sample_ohlc["low"], sample_ohlc["close"])
    assert (atr.dropna() >= 0).all()


def test_bollinger_upper_above_lower(sample_ohlc):
    mid, upper, lower = ind.bollinger_bands(sample_ohlc["close"], period=20)
    valid = upper.notna() & lower.notna()
    assert (upper[valid] >= lower[valid]).all()


def test_adx_bounded(sample_ohlc):
    adx_val, plus_di, minus_di = ind.adx(sample_ohlc["high"], sample_ohlc["low"], sample_ohlc["close"])
    valid = adx_val.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_true_vwap_matches_manual_calc(sample_ohlc):
    window = 10
    vwap = ind.true_vwap(
        sample_ohlc["high"], sample_ohlc["low"], sample_ohlc["close"], sample_ohlc["volume"], window=window
    )
    typical = (sample_ohlc["high"] + sample_ohlc["low"] + sample_ohlc["close"]) / 3.0
    manual = (typical * sample_ohlc["volume"]).iloc[50 - window : 50].sum() / sample_ohlc["volume"].iloc[
        50 - window : 50
    ].sum()
    assert np.isclose(vwap.iloc[49], manual)


def test_stochastic_rsi_bounded(sample_ohlc):
    k, d = ind.stochastic_rsi(sample_ohlc["close"])
    assert (k.dropna() >= 0).all() and (k.dropna() <= 100).all()
    assert (d.dropna() >= 0).all() and (d.dropna() <= 100).all()
