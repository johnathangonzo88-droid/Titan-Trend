"""Unit tests for src/data_layer/loader.py's format auto-detection and
data-quality reporting."""
import pandas as pd
import pytest

from src.data_layer.loader import load_ohlcv_csv


def test_loads_epoch_seconds_ohlc_without_volume(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "time,open,high,low,close\n"
        "1700000000,100,101,99,100.5\n"
        "1700000900,100.5,102,100,101.5\n"
        "1700001800,101.5,101.8,100.9,101.0\n"
    )
    df, report = load_ohlcv_csv(csv_path, "TEST")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 3
    assert report.has_volume is False
    assert isinstance(df.index, pd.DatetimeIndex)


def test_loads_real_volume_when_present(tmp_path):
    csv_path = tmp_path / "sample_vol.csv"
    csv_path.write_text(
        "time,open,high,low,close,volume\n"
        "1700000000,100,101,99,100.5,500\n"
        "1700000900,100.5,102,100,101.5,700\n"
    )
    df, report = load_ohlcv_csv(csv_path, "TEST")
    assert report.has_volume is True
    assert df["volume"].tolist() == [500.0, 700.0]


def test_empty_volume_column_treated_as_no_volume(tmp_path):
    csv_path = tmp_path / "sample_empty_vol.csv"
    csv_path.write_text(
        "time,open,high,low,close,volume\n"
        "1700000000,100,101,99,100.5,\n"
        "1700000900,100.5,102,100,101.5,\n"
    )
    df, report = load_ohlcv_csv(csv_path, "TEST")
    assert report.has_volume is False


def test_duplicate_timestamps_are_deduplicated(tmp_path):
    csv_path = tmp_path / "dupes.csv"
    csv_path.write_text(
        "time,open,high,low,close\n"
        "1700000000,100,101,99,100.5\n"
        "1700000000,100,101,99,100.9\n"  # duplicate timestamp
        "1700000900,100.5,102,100,101.5\n"
    )
    df, report = load_ohlcv_csv(csv_path, "TEST")
    assert len(df) == 2
    assert report.n_duplicate_timestamps == 1


def test_missing_required_column_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("time,open,high,close\n1700000000,100,101,100.5\n")
    with pytest.raises(ValueError):
        load_ohlcv_csv(csv_path, "TEST")


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_ohlcv_csv("does_not_exist.csv", "TEST")
