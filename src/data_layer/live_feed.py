"""
Extension point for a real-time data feed.

The bot ships wired to `CsvReplayDataSource` (src/telegram_bot/bot.py),
which replays the historical CSVs you provided so the whole pipeline can be
validated without a broker connection. To go live, implement a class here
with the same interface and pass it to `SignalBot(..., data_source_factory=...)`:

    class LiveDataSource:
        def __init__(self, symbol_cfg: dict): ...
        def latest_bars(self, symbol: str) -> tuple[pd.DataFrame, bool]:
            '''Return (canonical OHLCV dataframe, has_volume) with enough
            trailing history for indicator warm-up (>= 250 bars recommended
            given the longest lookback, EMA-200) plus the newest bar.'''

Common integrations to implement here:
  * A broker API with historical + streaming bars (Interactive Brokers,
    Tradovate, Rithmic, etc.) - typically REST for history + a websocket or
    polling loop for the newest bar.
  * A market data vendor (Polygon.io, Databento, etc.).
  * `yfinance` for a free/delayed feed, consistent with this project's
    existing ORB bot (see project notes) - note yfinance intraday history is
    limited to the last ~60 days, so it suits live polling better than
    long-history backtesting.

Whatever the source, the ONLY contract that matters to the rest of the
system is the canonical schema documented in src/data_layer/loader.py:
a DatetimeIndex named "datetime" and columns [open, high, low, close, volume].
"""
from __future__ import annotations

import pandas as pd


class NotImplementedLiveDataSource:
    """Placeholder that raises a clear error instead of silently doing
    nothing, if LIVE_DATA_SOURCE=broker is set before a real feed is wired up."""

    def __init__(self, symbol_cfg: dict):
        self.symbol_cfg = symbol_cfg

    def latest_bars(self, symbol: str) -> tuple[pd.DataFrame, bool]:
        raise NotImplementedError(
            f"No live data feed is configured for {symbol}. Implement a data source "
            f"class in src/data_layer/live_feed.py (see this file's docstring) and "
            f"pass it to SignalBot's data_source_factory."
        )
