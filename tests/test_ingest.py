from pathlib import Path
import pandas as pd
from python.prefect import flows


def test_ingest_sample_data(tmp_path, monkeypatch):
    # redirect data directory
    flows.DATA_DIR = tmp_path / "data"
    flows.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # stub dvc subprocess calls
    monkeypatch.setattr(flows.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(flows, "ensure_dvc_repo", lambda: None)

    # fake yfinance download
    class FakeYF:
        def download(self, ticker, start, end, interval, progress):
            index = pd.date_range("2020-01-01", periods=2, freq="D")
            df = pd.DataFrame({"Open": [1, 2], "High": [1, 2], "Low": [1, 2], "Close": [1, 2]}, index=index)
            return df

    monkeypatch.setattr(flows, "yf", FakeYF())

    path = flows.fetch_and_store.fn(
        ticker="TEST",
        start="2020-01-01",
        end="2020-01-02",
        freq="day",
    )
    assert path.exists()
