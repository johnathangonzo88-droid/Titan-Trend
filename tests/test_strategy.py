"""Unit tests for src/strategy/strategy.py and src/strategy/risk.py."""
import pandas as pd
import pytest

from src.strategy.risk import compute_risk_levels, position_size
from src.strategy.strategy import BUY, HOLD, SELL, generate_signal


STRATEGY_CFG = {
    "trend_ema_fast": 21,
    "trend_ema_slow": 50,
    "rsi_oversold": 35,
    "rsi_overbought": 65,
    "stoch_rsi_oversold": 20,
    "stoch_rsi_overbought": 80,
    "min_atr_pct": 0.03,
    "max_atr_pct": 3.0,
    "volume_proxy_z_threshold": 0.3,
    "atr_stop_multiplier": 1.5,
    "atr_tp1_multiplier": 2.25,
    "atr_tp2_multiplier": 4.0,
    "risk_per_trade_pct": 1.0,
    "min_risk_reward": 1.5,
}


def _base_row(**overrides) -> pd.Series:
    row = {
        "close": 100.0,
        "ema_21": 100.5,
        "ema_50": 99.0,
        "di_diff": 2.0,
        "rsi": 45.0,
        "stoch_rsi_k": 40.0,
        "stoch_rsi_d": 30.0,
        "macd_hist": 0.5,
        "macd_hist_rising": 1.0,
        "has_volume": False,
        "participation_proxy_z": 0.8,
        "atr_pct": 0.15,
        "atr": 1.0,
        "regime": "trending",
    }
    row.update(overrides)
    return pd.Series(row)


def test_compute_risk_levels_buy():
    levels = compute_risk_levels("BUY", entry_price=100.0, atr_value=2.0, point_value=50.0,
                                  stop_multiplier=1.5, tp1_multiplier=2.25, tp2_multiplier=4.0)
    assert levels.stop_loss == 100.0 - 1.5 * 2.0
    assert levels.take_profit_1 == 100.0 + 2.25 * 2.0
    assert levels.take_profit_2 == 100.0 + 4.0 * 2.0
    assert levels.reward_to_risk_tp1 == pytest.approx(2.25 / 1.5)


def test_compute_risk_levels_sell_mirrors_buy():
    levels = compute_risk_levels("SELL", entry_price=100.0, atr_value=2.0, point_value=50.0)
    assert levels.stop_loss > 100.0
    assert levels.take_profit_1 < 100.0


def test_compute_risk_levels_invalid_direction():
    with pytest.raises(ValueError):
        compute_risk_levels("WAIT", 100.0, 2.0, 50.0)


def test_position_size_respects_risk_budget():
    size = position_size(account_equity=50000, risk_per_trade_pct=1.0, risk_per_contract=100.0)
    assert size == 5  # $500 budget / $100 risk per contract


def test_position_size_zero_risk_returns_zero():
    assert position_size(50000, 1.0, 0.0) == 0


def test_generate_signal_buy_on_full_confluence():
    row = _base_row()
    result = generate_signal(row, STRATEGY_CFG, point_value=50.0)
    assert result.action == BUY
    assert result.risk_levels is not None
    assert result.risk_levels.stop_loss < row["close"]


def test_generate_signal_hold_when_ranging():
    row = _base_row(regime="ranging")
    result = generate_signal(row, STRATEGY_CFG, point_value=50.0)
    assert result.action == HOLD


def test_generate_signal_hold_when_volatile():
    row = _base_row(regime="volatile")
    result = generate_signal(row, STRATEGY_CFG, point_value=50.0)
    assert result.action == HOLD


def test_generate_signal_hold_when_atr_too_low():
    row = _base_row(atr_pct=0.01)
    result = generate_signal(row, STRATEGY_CFG, point_value=50.0)
    assert result.action == HOLD


def test_generate_signal_sell_on_bearish_confluence():
    row = _base_row(
        ema_21=99.0, ema_50=100.5, di_diff=-2.0,
        rsi=55.0, stoch_rsi_k=30.0, stoch_rsi_d=40.0,
        macd_hist=-0.5, macd_hist_rising=0.0,
    )
    result = generate_signal(row, STRATEGY_CFG, point_value=50.0)
    assert result.action == SELL


def test_generate_signal_ml_veto_flips_buy_to_hold():
    row = _base_row()
    result = generate_signal(row, STRATEGY_CFG, point_value=50.0, ml_probability=0.1)
    assert result.action == HOLD


def test_generate_signal_ml_confirmation_keeps_buy():
    row = _base_row()
    result = generate_signal(row, STRATEGY_CFG, point_value=50.0, ml_probability=0.8)
    assert result.action == BUY
