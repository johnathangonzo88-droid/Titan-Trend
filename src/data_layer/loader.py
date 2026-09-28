"""
Data loading and format auto-detection.

Handles the messy reality of "historical data I provide": different column
names, epoch vs. string timestamps, extra empty indicator columns exported
from charting platforms, and the presence/absence of a volume column.

The loader's job is to always hand the rest of the pipeline a single
canonical schema:

    index: pandas.DatetimeIndex (UTC), name="datetime"
    columns: open, high, low, close, volume  (volume is NaN if unavailable)

Everything downstream (features, strategy, ML, backtest) depends only on
this canonical schema, never on the raw file's quirks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

CANONICAL_COLUMNS = ["open", "high", "low", "close", "volume"]

# Column name aliases we recognize across common export formats
# (TradingView, MetaTrader, Interactive Brokers, generic OHLCV CSVs, ccxt, etc.)
_COLUMN_ALIASES = {
    "open": {"open", "o", "Open", "OPEN"},
    "high": {"high", "h", "High", "HIGH"},
    "low": {"low", "l", "Low", "LOW"},
    "close": {"close", "c", "Close", "CLOSE", "adj close", "adj_close"},
    "volume": {"volume", "vol", "Volume", "VOLUME", "v", "tick_volume", "Volume MA"},
    "time": {"time", "timestamp", "date", "datetime", "Date", "Time", "Timestamp", "Gmt time"},
}


@dataclass
class DataQualityReport:
    """Summary of issues found while loading a file, surfaced to the caller
    instead of silently "fixing" data the user should know about."""

    symbol: str
    n_rows: int
    start: pd.Timestamp
    end: pd.Timestamp
    has_volume: bool
    n_duplicate_timestamps: int
    n_ohlc_violations: int
    n_gaps_detected: int
    inferred_bar_seconds: float
    notes: list

    def summary(self) -> str:
        lines = [
            f"[{self.symbol}] {self.n_rows} bars from {self.start} to {self.end}",
            f"  bar size: ~{self.inferred_bar_seconds:.0f}s | volume present: {self.has_volume}",
            f"  duplicate timestamps: {self.n_duplicate_timestamps} | "
            f"OHLC violations: {self.n_ohlc_violations} | session gaps: {self.n_gaps_detected}",
        ]
        for n in self.notes:
            lines.append(f"  NOTE: {n}")
        return "\n".join(lines)


def _find_column(df: pd.DataFrame, canonical_name: str) -> Optional[str]:
    aliases = _COLUMN_ALIASES[canonical_name]
    for col in df.columns:
        if col in aliases or col.strip().lower() in {a.lower() for a in aliases}:
            return col
    return None


def _parse_timestamp_column(series: pd.Series) -> pd.DatetimeIndex:
    """Auto-detect whether a timestamp column is epoch seconds, epoch
    milliseconds, or an ISO/string date, and parse accordingly."""
    if np.issubdtype(series.dtype, np.number):
        sample = series.dropna().iloc[0]
        # Heuristic: epoch seconds are ~1.7e9 today; milliseconds ~1.7e12.
        if sample > 1e12:
            return pd.to_datetime(series, unit="ms", utc=True)
        else:
            return pd.to_datetime(series, unit="s", utc=True)
    return pd.to_datetime(series, utc=True, errors="coerce")


def load_ohlcv_csv(path: str | Path, symbol: str) -> tuple[pd.DataFrame, DataQualityReport]:
    """Load a single OHLC(V) CSV, auto-detecting its layout.

    Returns the canonicalized dataframe plus a DataQualityReport so callers
    (and the EDA script) can decide whether the data is fit for purpose
    rather than training silently on bad input.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such data file: {path}")

    raw = pd.read_csv(path)
    notes: list = []

    time_col = _find_column(raw, "time")
    if time_col is None:
        raise ValueError(
            f"Could not detect a timestamp column in {path.name}. "
            f"Columns found: {list(raw.columns)}"
        )

    dt_index = _parse_timestamp_column(raw[time_col])

    out = pd.DataFrame(index=dt_index)
    out.index.name = "datetime"

    for canonical in ["open", "high", "low", "close"]:
        col = _find_column(raw, canonical)
        if col is None:
            raise ValueError(f"Missing required column '{canonical}' in {path.name}")
        out[canonical] = raw[col].values

    vol_col = _find_column(raw, "volume")
    has_volume = False
    if vol_col is not None:
        vol_values = raw[vol_col]
        # A column that's entirely NaN or entirely zero isn't real volume data
        # even if a "volume"-like column exists (common in some chart exports).
        if vol_values.notna().any() and (vol_values.fillna(0) != 0).any():
            out["volume"] = vol_values.values
            has_volume = True
        else:
            notes.append(
                f"A '{vol_col}' column exists but is empty/all-zero; treating as no volume data."
            )
            out["volume"] = np.nan
    else:
        out["volume"] = np.nan
        notes.append("No volume column found in source file; volume-based features will use proxies.")

    # Drop rows with any NaN in required OHLC fields
    n_before = len(out)
    out = out.dropna(subset=["open", "high", "low", "close"])
    if len(out) < n_before:
        notes.append(f"Dropped {n_before - len(out)} rows with missing OHLC values.")

    # Sort and deduplicate on timestamp (keep first occurrence)
    out = out.sort_index()
    n_dupes = int(out.index.duplicated().sum())
    if n_dupes > 0:
        out = out[~out.index.duplicated(keep="first")]
        notes.append(f"Removed {n_dupes} duplicate timestamps.")

    # OHLC sanity: high must be >= max(open, close), low <= min(open, close)
    violations = (out["high"] < out[["open", "close"]].max(axis=1)) | (
        out["low"] > out[["open", "close"]].min(axis=1)
    )
    n_violations = int(violations.sum())
    if n_violations > 0:
        notes.append(
            f"{n_violations} bars violate OHLC bounds (high/low don't contain open/close); "
            f"kept as-is but flagged for review."
        )

    # Infer nominal bar spacing as the mode of time diffs (robust to session gaps)
    diffs = out.index.to_series().diff().dropna().dt.total_seconds()
    inferred_seconds = float(diffs.mode().iloc[0]) if len(diffs) else float("nan")
    # A "gap" is any diff more than 2x the nominal bar size (session breaks, weekends)
    n_gaps = int((diffs > inferred_seconds * 2).sum()) if not np.isnan(inferred_seconds) else 0

    report = DataQualityReport(
        symbol=symbol,
        n_rows=len(out),
        start=out.index.min(),
        end=out.index.max(),
        has_volume=has_volume,
        n_duplicate_timestamps=n_dupes,
        n_ohlc_violations=n_violations,
        n_gaps_detected=n_gaps,
        inferred_bar_seconds=inferred_seconds,
        notes=notes,
    )

    logger.info("Loaded %s: %d bars (%s -> %s)", symbol, len(out), report.start, report.end)
    for n in notes:
        logger.warning("[%s] %s", symbol, n)

    return out[CANONICAL_COLUMNS], report


def load_all_symbols(symbol_configs: list) -> dict:
    """Load every symbol listed in config.yaml's `symbols` section.

    Returns {symbol_name: (dataframe, DataQualityReport)}.
    """
    results = {}
    for cfg in symbol_configs:
        df, report = load_ohlcv_csv(cfg["raw_file"], cfg["name"])
        results[cfg["name"]] = (df, report)
    return results
