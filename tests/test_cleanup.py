from datetime import datetime, timedelta
import os
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from daxai.prefect import cleanup as c


def test_drop_unreferenced_aggregates_old_minutes(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "minute"
    data_dir.mkdir(parents=True)
    p = data_dir / "x.parquet"
    df = pd.DataFrame({"Open": [1], "High": [1], "Low": [1], "Close": [1]}, index=pd.date_range("2020-01-01", periods=1, freq="T"))
    p.write_bytes(b"dummy")
    old = datetime.now() - timedelta(days=100)
    p.touch()
    os.utime(p, (old.timestamp(), old.timestamp()))

    monkeypatch.setattr(c, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(c, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(c, "_resample_5min", lambda df: df)
    monkeypatch.setattr(c.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(c.pd, "read_parquet", lambda x: df)
    monkeypatch.setattr(c, "load_config", lambda name: {"aggregate_minute_after_days": 90})
    import logging
    monkeypatch.setattr(c, "get_run_logger", lambda: logging.getLogger("test"))

    called = {"aggregated": False}
    def resample(df):
        called["aggregated"] = True
        return df
    monkeypatch.setattr(c, "_resample_5min", resample)

    c.drop_unreferenced_parquet.fn()
    assert called["aggregated"], "Old minute file not aggregated"
