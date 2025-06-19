from pathlib import Path

import pandas as pd
import pytest
import yfinance as yf

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def yf_daily_df() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE_DIR / "yf_daily.csv", index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


@pytest.fixture(scope="session")
def yf_minute_df() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE_DIR / "yf_minute.csv", index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


@pytest.fixture(autouse=True)
def patch_yfinance(monkeypatch: pytest.MonkeyPatch, yf_daily_df: pd.DataFrame, yf_minute_df: pd.DataFrame):
    def fake_download(ticker, start=None, end=None, interval="1d", auto_adjust=False, progress=False, threads=False):
        if interval == "1m":
            return yf_minute_df.copy()
        return yf_daily_df.copy()

    monkeypatch.setattr(yf, "download", fake_download)
    return fake_download
