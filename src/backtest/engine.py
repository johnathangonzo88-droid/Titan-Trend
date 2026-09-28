"""
Event-driven backtesting engine.

Walks the featured dataframe bar by bar (never vectorized in a way that
could peek ahead), calls the strategy for a signal only when flat, and once
in a position, manages it exactly the way live_signal.py's risk levels
describe: a 50% scale-out at Take Profit 1 with the stop moved to
breakeven, the remainder targeting Take Profit 2, and a full stop-loss exit
possible at any time.

Conservative same-bar tie-break: if a single bar's high/low range would hit
both the stop and a target, the stop is assumed to fill first. This avoids
the classic backtesting overoptimism bug of always giving the trade the
benefit of the doubt.

ML probabilities, when supplied, MUST already be out-of-sample (i.e. built
by src.ml.walkforward's walk-forward folds, never the full-data refit
model) - the engine does not enforce this itself, so callers are
responsible for passing leak-free probabilities (see scripts/backtest.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.risk import compute_risk_levels, position_size
from src.strategy.strategy import BUY, HOLD, SELL, generate_signal
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OpenPosition:
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    contracts: int
    remaining_contracts: int
    tp1_hit: bool = False
    reasons: Optional[list] = None


class BacktestEngine:
    def __init__(
        self,
        features_df: pd.DataFrame,
        strategy_cfg: dict,
        backtest_cfg: dict,
        point_value: float,
        tick_size: float,
        ml_proba: Optional[pd.Series] = None,
    ):
        self.df = features_df
        self.strategy_cfg = strategy_cfg
        self.backtest_cfg = backtest_cfg
        self.point_value = point_value
        self.tick_size = tick_size
        self.ml_proba = ml_proba
        self.slippage_points = backtest_cfg.get("slippage_ticks", 1) * tick_size
        self.commission = backtest_cfg.get("commission_per_trade", 0.0)

    def _current_ml_proba(self, timestamp) -> Optional[float]:
        if self.ml_proba is None:
            return None
        val = self.ml_proba.get(timestamp, np.nan)
        return None if pd.isna(val) else float(val)

    def _close_position(
        self, pos: OpenPosition, exit_time, exit_price: float, contracts: int, equity: float, reason: str
    ) -> tuple[dict, float]:
        sign = 1 if pos.direction == BUY else -1
        exit_price_after_slippage = exit_price - sign * self.slippage_points
        pnl_points = sign * (exit_price_after_slippage - pos.entry_price)
        pnl_dollars = pnl_points * self.point_value * contracts - self.commission

        trade = {
            "entry_time": pos.entry_time,
            "exit_time": exit_time,
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "exit_price": exit_price_after_slippage,
            "contracts": contracts,
            "pnl": pnl_dollars,
            "exit_reason": reason,
            "bars_held": None,  # filled in by caller if needed
        }
        return trade, equity + pnl_dollars

    def run(self, initial_equity: Optional[float] = None) -> tuple[pd.DataFrame, pd.Series]:
        equity = initial_equity if initial_equity is not None else self.backtest_cfg["initial_equity"]
        equity_curve = []
        trades = []
        position: Optional[OpenPosition] = None
        bars_in_trade = 0

        rows = list(self.df.itertuples(index=True))

        for row in rows:
            row_dict = row._asdict()
            timestamp = row_dict["Index"]
            high, low = row_dict["high"], row_dict["low"]
            floating_pnl = 0.0

            if position is not None:
                bars_in_trade += 1
                sign = 1 if position.direction == BUY else -1

                # Conservative same-bar ordering: stop checked before targets.
                stop_hit = (low <= position.stop_loss) if position.direction == BUY else (high >= position.stop_loss)
                tp1_hit = (high >= position.take_profit_1) if position.direction == BUY else (low <= position.take_profit_1)
                tp2_hit = (high >= position.take_profit_2) if position.direction == BUY else (low <= position.take_profit_2)

                if stop_hit:
                    trade, equity = self._close_position(
                        position, timestamp, position.stop_loss, position.remaining_contracts,
                        equity, "stop_loss" if not position.tp1_hit else "stop_loss_breakeven",
                    )
                    trade["bars_held"] = bars_in_trade
                    trades.append(trade)
                    position = None
                    bars_in_trade = 0
                elif tp2_hit:
                    trade, equity = self._close_position(
                        position, timestamp, position.take_profit_2, position.remaining_contracts,
                        equity, "take_profit_2",
                    )
                    trade["bars_held"] = bars_in_trade
                    trades.append(trade)
                    position = None
                    bars_in_trade = 0
                elif tp1_hit and not position.tp1_hit and position.remaining_contracts > 1:
                    # Scale out 50%, move stop to breakeven, keep riding the rest to TP2.
                    scale_out = position.remaining_contracts // 2
                    trade, equity = self._close_position(
                        position, timestamp, position.take_profit_1, scale_out, equity, "take_profit_1_partial",
                    )
                    trade["bars_held"] = bars_in_trade
                    trades.append(trade)
                    position.remaining_contracts -= scale_out
                    position.stop_loss = position.entry_price
                    position.tp1_hit = True
                elif tp1_hit and not position.tp1_hit:
                    # Only 1 contract total - take full profit at TP1 rather than
                    # leave a fractional position open.
                    trade, equity = self._close_position(
                        position, timestamp, position.take_profit_1, position.remaining_contracts,
                        equity, "take_profit_1_full",
                    )
                    trade["bars_held"] = bars_in_trade
                    trades.append(trade)
                    position = None
                    bars_in_trade = 0
                else:
                    # Mark-to-market unrealized P&L for the equity curve.
                    close = row_dict["close"]
                    floating_pnl = sign * (close - position.entry_price) * self.point_value * position.remaining_contracts

            if position is None and not pd.isna(row_dict.get("adx", np.nan)):
                signal = generate_signal(
                    row=pd.Series(row_dict),
                    cfg=self.strategy_cfg,
                    point_value=self.point_value,
                    ml_probability=self._current_ml_proba(timestamp),
                )
                if signal.action in (BUY, SELL) and signal.risk_levels is not None:
                    contracts = position_size(
                        account_equity=equity,
                        risk_per_trade_pct=self.strategy_cfg["risk_per_trade_pct"],
                        risk_per_contract=signal.risk_levels.risk_per_contract,
                    )
                    if contracts >= 1:
                        position = OpenPosition(
                            direction=signal.action,
                            entry_time=timestamp,
                            entry_price=signal.risk_levels.entry,
                            stop_loss=signal.risk_levels.stop_loss,
                            take_profit_1=signal.risk_levels.take_profit_1,
                            take_profit_2=signal.risk_levels.take_profit_2,
                            contracts=contracts,
                            remaining_contracts=contracts,
                            reasons=signal.reasons,
                        )
                        bars_in_trade = 0

            equity_curve.append(equity + floating_pnl)

        trades_df = pd.DataFrame(trades)
        equity_series = pd.Series(equity_curve, index=self.df.index, name="equity")

        starting_equity = initial_equity if initial_equity is not None else self.backtest_cfg["initial_equity"]
        logger.info(
            "Backtest complete: %d trades, final equity $%.2f (started $%.2f)",
            len(trades_df), equity_series.iloc[-1] if len(equity_series) else equity,
            starting_equity,
        )
        return trades_df, equity_series
