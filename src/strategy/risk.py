"""
Risk management: stop loss / take profit levels and position sizing.

Kept separate from the entry/exit signal logic in strategy.py so risk rules
can be tuned or unit-tested independently of "when do we enter".
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskLevels:
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_per_contract: float          # $ risked per contract (point risk * point value)
    reward_to_risk_tp1: float
    reward_to_risk_tp2: float


def compute_risk_levels(
    direction: str,
    entry_price: float,
    atr_value: float,
    point_value: float,
    stop_multiplier: float = 1.5,
    tp1_multiplier: float = 2.0,
    tp2_multiplier: float = 3.5,
) -> RiskLevels:
    """Compute ATR-based stop loss and two take-profit levels.

    ATR-based (rather than fixed-tick) stops adapt to current volatility:
    wider stops in choppy/volatile regimes, tighter in quiet ones, which is
    what the regime detector and volatility filter are for upstream.
    """
    if direction not in ("BUY", "SELL"):
        raise ValueError(f"direction must be BUY or SELL, got {direction!r}")

    sign = 1 if direction == "BUY" else -1

    stop_loss = entry_price - sign * atr_value * stop_multiplier
    take_profit_1 = entry_price + sign * atr_value * tp1_multiplier
    take_profit_2 = entry_price + sign * atr_value * tp2_multiplier

    risk_points = abs(entry_price - stop_loss)
    reward_1_points = abs(take_profit_1 - entry_price)
    reward_2_points = abs(take_profit_2 - entry_price)

    return RiskLevels(
        entry=entry_price,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        risk_per_contract=risk_points * point_value,
        reward_to_risk_tp1=reward_1_points / risk_points if risk_points > 0 else 0.0,
        reward_to_risk_tp2=reward_2_points / risk_points if risk_points > 0 else 0.0,
    )


def position_size(
    account_equity: float,
    risk_per_trade_pct: float,
    risk_per_contract: float,
    max_contracts: int = 20,
) -> int:
    """Number of contracts to trade so that a stop-out risks at most
    `risk_per_trade_pct`% of account equity.

    This is a recommendation, not an order-execution instruction - the bot
    reports it, the trader (or an execution layer the user builds) decides.
    """
    if risk_per_contract <= 0:
        return 0
    dollar_risk_budget = account_equity * (risk_per_trade_pct / 100.0)
    contracts = int(dollar_risk_budget // risk_per_contract)
    return max(0, min(contracts, max_contracts))
