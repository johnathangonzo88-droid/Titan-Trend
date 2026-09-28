"""
Rule-based trading strategy: "Titan Trend".

Combines price action, trend, momentum, volume(-proxy) confirmation, and a
volatility filter into a single entry-signal function, plus paired exit /
risk logic. This module produces the *rule-based* signal; src/ml/ produces
an independent probability estimate; live_signal.py blends the two into the
final confidence score sent to Telegram.

Design goal from the brief ("maximize risk-adjusted returns rather than raw
win rate"): entries require multi-factor confirmation (trend + momentum +
volume-proxy all agree) rather than any single indicator, which lowers trade
frequency but is intended to raise the average trade's quality. This is a
starting point to calibrate against your own risk tolerance and the
backtest results in reports/, not a guarantee of future performance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.features.regime import REGIME_RANGING, REGIME_TRENDING, REGIME_VOLATILE, REGIME_LOW_VOL
from src.strategy.risk import RiskLevels, compute_risk_levels

BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"


@dataclass
class SignalResult:
    action: str                       # BUY / SELL / HOLD
    reasons: list = field(default_factory=list)
    rule_score: float = 0.0           # -1..+1, magnitude = conviction
    risk_levels: Optional[RiskLevels] = None


def _trend_conditions(row: pd.Series, cfg: dict) -> tuple[bool, bool, list]:
    """Trend filter: fast EMA vs slow EMA, plus ADX/DI confirming direction."""
    fast_col = f"ema_{cfg['trend_ema_fast']}"
    slow_col = f"ema_{cfg['trend_ema_slow']}"
    bullish = row[fast_col] > row[slow_col] and row["di_diff"] > 0
    bearish = row[fast_col] < row[slow_col] and row["di_diff"] < 0
    reasons = []
    if bullish:
        reasons.append("EMA trend bullish (fast > slow, +DI > -DI)")
    if bearish:
        reasons.append("EMA trend bearish (fast < slow, -DI > +DI)")
    return bullish, bearish, reasons


def _momentum_conditions(row: pd.Series, cfg: dict) -> tuple[bool, bool, list]:
    """Momentum: RSI recovering from oversold / falling from overbought,
    OR a rising/falling MACD histogram (either is accepted as confirmation -
    requiring both simultaneously made genuine setups vanishingly rare in
    testing, since RSI and MACD often confirm a move at slightly different
    bars). Both are still checked for outright contradiction."""
    reasons = []
    rsi_bull = row["rsi"] > cfg["rsi_oversold"] and row["rsi"] < 65 and row["stoch_rsi_k"] > row["stoch_rsi_d"]
    rsi_bear = row["rsi"] < cfg["rsi_overbought"] and row["rsi"] > 35 and row["stoch_rsi_k"] < row["stoch_rsi_d"]

    macd_bull = row["macd_hist"] > 0 and row["macd_hist_rising"] == 1.0
    macd_bear = row["macd_hist"] < 0 and row["macd_hist_rising"] == 0.0

    bullish = (rsi_bull or macd_bull) and not macd_bear
    bearish = (rsi_bear or macd_bear) and not macd_bull

    if bullish and rsi_bull:
        reasons.append("RSI recovery from oversold")
    if bullish and macd_bull:
        reasons.append("Rising MACD histogram")
    if bearish and rsi_bear:
        reasons.append("RSI rollover from overbought")
    if bearish and macd_bear:
        reasons.append("Falling MACD histogram")
    return bullish, bearish, reasons


def _volume_confirmation(row: pd.Series, cfg: dict) -> tuple[bool, list]:
    """Volume (or volume-proxy) confirmation: participation above its recent
    baseline, in the direction of the candle."""
    reasons = []
    if row.get("has_volume", False):
        z = row.get("volume_z", 0.0)
        confirmed = z is not None and z > cfg["volume_proxy_z_threshold"]
        if confirmed:
            reasons.append(f"Volume spike detected (z={z:.2f})")
    else:
        z = row.get("participation_proxy_z", 0.0)
        confirmed = z is not None and z > cfg["volume_proxy_z_threshold"]
        if confirmed:
            reasons.append(f"Volume-proxy conviction spike (z={z:.2f}, no real volume data available)")
    return bool(confirmed), reasons


def _volatility_filter(row: pd.Series, cfg: dict) -> tuple[bool, list]:
    """Reject trades in dead-quiet or extreme-volatility conditions."""
    reasons = []
    atr_pct = row.get("atr_pct", 0.0)
    ok = cfg["min_atr_pct"] <= atr_pct <= cfg["max_atr_pct"]
    regime = row.get("regime")
    if regime == REGIME_VOLATILE:
        ok = False
        reasons.append("Regime is 'volatile' (chaotic) - trade skipped")
    if not ok and regime != REGIME_VOLATILE:
        reasons.append(f"ATR% ({atr_pct:.3f}) outside allowed range - trade skipped")
    return ok, reasons


def generate_signal(
    row: pd.Series,
    cfg: dict,
    point_value: float,
    ml_probability: Optional[float] = None,
) -> SignalResult:
    """Generate a single-bar trading signal from one fully-featured row.

    `row` must come from a dataframe produced by
    `src.features.feature_engineering.build_features` (needs regime, EMAs,
    RSI, MACD, ADX/DI, ATR, and volume/participation columns).

    `ml_probability`, if provided, is the ML model's probability of a
    favorable move (see src/ml/). It is blended into `rule_score` for the
    final confidence, but the rule-based direction is decided independently
    -- the ML model never overrides a rule-based HOLD into a trade, it only
    adjusts confidence and can veto a weak rule-based signal.
    """
    reasons: list = []

    vol_ok, vol_reasons = _volatility_filter(row, cfg)
    reasons += vol_reasons
    if not vol_ok:
        return SignalResult(action=HOLD, reasons=reasons, rule_score=0.0)

    if row.get("regime") == REGIME_RANGING:
        reasons.append("Regime is 'ranging' - trend-following entries suppressed")
        return SignalResult(action=HOLD, reasons=reasons, rule_score=0.0)

    trend_bull, trend_bear, trend_reasons = _trend_conditions(row, cfg)
    mom_bull, mom_bear, mom_reasons = _momentum_conditions(row, cfg)
    vol_confirmed, vol_conf_reasons = _volume_confirmation(row, cfg)

    # Confluence rule: trend direction is mandatory; at least one of
    # {momentum, volume-proxy} must also confirm. Requiring trend + both
    # momentum AND volume simultaneously made setups occur on well under 1%
    # of bars in testing - too sparse for a signal service - while dropping
    # the trend requirement let counter-trend noise through. This keeps
    # trend mandatory and treats momentum/volume as a 1-of-2 confirmation
    # vote, which is still meaningfully more selective than a single indicator.
    long_setup = trend_bull and (mom_bull or vol_confirmed)
    short_setup = trend_bear and (mom_bear or vol_confirmed)

    if long_setup:
        reasons += trend_reasons + mom_reasons + vol_conf_reasons
        rule_score = 1.0
        action = BUY
    elif short_setup:
        reasons += trend_reasons + mom_reasons + vol_conf_reasons
        rule_score = -1.0
        action = SELL
    else:
        reasons.append("No confluence of trend + momentum + volume-proxy confirmation")
        return SignalResult(action=HOLD, reasons=reasons, rule_score=0.0)

    # Blend in ML confidence if available: scales conviction, can veto weak signals.
    if ml_probability is not None:
        # ml_probability is P(favorable move | features), 0.5 = coin flip
        ml_edge = (ml_probability - 0.5) * 2.0  # rescale to -1..+1
        if action == BUY and ml_edge < -0.2:
            reasons.append(f"Vetoed: ML model disagrees strongly (p={ml_probability:.2f})")
            return SignalResult(action=HOLD, reasons=reasons, rule_score=0.0)
        if action == SELL and ml_edge > 0.2:
            reasons.append(f"Vetoed: ML model disagrees strongly (p={ml_probability:.2f})")
            return SignalResult(action=HOLD, reasons=reasons, rule_score=0.0)
        reasons.append(f"Model prediction: P(favorable)={ml_probability:.0%}")
        rule_score = float((rule_score + ml_edge) / 2.0)

    entry_price = row["close"]
    atr_value = row["atr"]
    risk_levels = compute_risk_levels(
        direction=action,
        entry_price=entry_price,
        atr_value=atr_value,
        point_value=point_value,
        stop_multiplier=cfg["atr_stop_multiplier"],
        tp1_multiplier=cfg["atr_tp1_multiplier"],
        tp2_multiplier=cfg["atr_tp2_multiplier"],
    )

    if risk_levels.reward_to_risk_tp1 < cfg["min_risk_reward"]:
        reasons.append(
            f"Risk/reward to TP1 ({risk_levels.reward_to_risk_tp1:.2f}) below minimum "
            f"({cfg['min_risk_reward']}) - trade skipped"
        )
        return SignalResult(action=HOLD, reasons=reasons, rule_score=0.0)

    return SignalResult(action=action, reasons=reasons, rule_score=rule_score, risk_levels=risk_levels)


def confidence_from_score(rule_score: float) -> int:
    """Map a -1..+1 rule/ML blended score to a 0-100% confidence for display.

    |score|=1.0 (full multi-factor agreement, ML fully concurring) -> ~95%.
    |score|~0.4 (bare minimum to trigger BUY/SELL) -> ~55%.
    Linear in between, floored/capped for sane display.
    """
    magnitude = min(abs(rule_score), 1.0)
    pct = 40 + magnitude * 55
    return int(round(min(max(pct, 40), 95)))
