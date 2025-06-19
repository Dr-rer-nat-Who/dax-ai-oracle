import pandas as pd
import pytest
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
        def download(self, ticker, start, end, interval, auto_adjust, progress, threads=False):
            index = pd.date_range("2020-01-01", periods=2, freq="D", tz="UTC")
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

def test_fetch_and_store_minute_chunks(tmp_path, monkeypatch):
    flows.DATA_DIR = tmp_path / "data"
    flows.DATA_DIR.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(flows.subprocess, "run", lambda *a, **k: None)

    calls = []

    def fake_download(ticker, start, end, interval, auto_adjust, progress, threads=False):
        calls.append((start, end))
        idx = pd.date_range(start, periods=1, freq="T", tz="UTC")
        return pd.DataFrame({"Open": [1], "High": [1], "Low": [1], "Close": [1]}, index=idx)

    monkeypatch.setattr(flows, "yf", type("_YF", (), {"download": staticmethod(fake_download)}))
    monkeypatch.setattr(flows.pd.Timestamp, "utcnow", staticmethod(lambda: pd.Timestamp("2024-02-01")))

    dest = flows.DATA_DIR / "raw" / "minute"
    dest.mkdir(parents=True, exist_ok=True)
    existing = pd.DataFrame(
        {"Open": [1], "High": [1], "Low": [1], "Close": [1]},
        index=pd.date_range("2020-01-01", periods=1, freq="T"),
    )
    existing.to_parquet(dest / "TEST.parquet")

    path = flows.fetch_and_store.fn(
        ticker="TEST",
        start="2020-01-01",
        end="2020-02-15",
        freq="minute",
    )

    assert path.exists()
    assert len(calls) >= 2
    assert pd.Timestamp(calls[0][0]).tz_localize(None) == existing.index.max() + pd.Timedelta(minutes=1)
    df = pd.read_parquet(path)
    assert df.index.tz is None
    for s, e in calls:
        assert pd.Timestamp(e) - pd.Timestamp(s) <= pd.Timedelta(days=8)


def test_fetch_and_store_minute_start_cutoff(tmp_path, monkeypatch):
    flows.DATA_DIR = tmp_path / "data"
    flows.DATA_DIR.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(flows.subprocess, "run", lambda *a, **k: None)

    starts = []

    def fake_download(ticker, start, end, interval, auto_adjust, progress):
        starts.append(start)
        idx = pd.date_range(start, periods=1, freq="T", tz="UTC")
        return pd.DataFrame({"Open": [1], "High": [1], "Low": [1], "Close": [1]}, index=idx)

    monkeypatch.setattr(flows, "yf", type("_YF", (), {"download": staticmethod(fake_download)}))
    monkeypatch.setattr(flows.pd.Timestamp, "utcnow", staticmethod(lambda: pd.Timestamp("2024-05-30")))

    path = flows.fetch_and_store.fn(
        ticker="TEST",
        start="2024-03-01",
        end="2024-05-31",
        freq="minute",
    )

    assert path.exists()
    cutoff = pd.Timestamp("2024-04-30", tz="UTC")
    ts = pd.Timestamp(starts[0])
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    assert ts >= cutoff


def test_fetch_and_store_handles_missing_error(tmp_path, monkeypatch):
    flows.DATA_DIR = tmp_path / "data"
    flows.DATA_DIR.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(flows.subprocess, "run", lambda *a, **k: None)

    dest = flows.DATA_DIR / "raw" / "day"
    dest.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"Open": [1]}, index=pd.date_range("2020-01-01", periods=1, freq="D"))
    p = dest / "TEST.parquet"
    df.to_parquet(p)

    def raise_missing(*a, **k):
        raise flows.YFPricesMissingError("missing")

    monkeypatch.setattr(flows, "yf", type("_YF", (), {"download": staticmethod(raise_missing)}))

    with pytest.raises(flows.YFPricesMissingError):
        flows.fetch_and_store.fn("TEST", "2020-01-01", "2020-01-02", "day")


def test_fetch_and_store_handles_generic_error(tmp_path, monkeypatch):
    flows.DATA_DIR = tmp_path / "data"
    flows.DATA_DIR.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(flows.subprocess, "run", lambda *a, **k: None)

    def raise_error(*a, **k):
        raise RuntimeError("fail")

    monkeypatch.setattr(flows, "yf", type("_YF", (), {"download": staticmethod(raise_error)}))

    path = flows.fetch_and_store.fn("TEST", "2020-01-01", "2020-01-02", "day")

    expect = flows.DATA_DIR / "raw" / "day" / "TEST.parquet"
    assert path == expect
    assert not path.exists()


def test_fetch_and_store_empty_df_raises(tmp_path, monkeypatch):
    flows.DATA_DIR = tmp_path / "data"
    flows.DATA_DIR.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(flows.subprocess, "run", lambda *a, **k: None)

    def empty_download(*a, **k):
        return pd.DataFrame()

    monkeypatch.setattr(flows, "yf", type("_YF", (), {"download": staticmethod(empty_download)}))

    with pytest.raises(flows.YFPricesMissingError):
        flows.fetch_and_store.fn("TEST", "2020-01-01", "2020-01-02", "day")

