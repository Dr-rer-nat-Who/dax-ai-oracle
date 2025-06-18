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


def test_ingest_per_freq_ranges(tmp_path, monkeypatch):
    flows.DATA_DIR = tmp_path / "data"
    flows.DATA_DIR.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(flows.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(flows, "ensure_dvc_repo", lambda: None)
    monkeypatch.setattr(flows, "remove_checkpoints", lambda: None, raising=False)

    calls: list[tuple[str, str, str]] = []

    def fake_fetch(ticker: str, start: str, end: str, freq: str):
        calls.append((start, end, freq))
        dest = flows.DATA_DIR / "raw" / freq
        dest.mkdir(parents=True, exist_ok=True)
        p = dest / f"{ticker}.parquet"
        p.write_text("x")
        return p

    monkeypatch.setattr(flows, "fetch_and_store", fake_fetch)

    cfg = {
        "start_date": "2020-01-01",
        "end_date": "2020-01-10",
        "tickers": ["TEST"],
        "frequencies": ["day", "hour"],
        "date_ranges": {"day": {"start_date": "2021-01-01", "end_date": "2021-01-10"}},
    }

    flows.ingest.fn("all", config=cfg)

    assert calls == [
        ("2021-01-01", "2021-01-10", "day"),
        ("2020-01-01", "2020-01-10", "hour"),
    ]
