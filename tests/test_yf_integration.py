import socket
import pandas as pd
import pytest
import sys
import importlib
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if isinstance(sys.modules.get("yfinance"), types.SimpleNamespace):
    del sys.modules["yfinance"]
yf = importlib.import_module("yfinance")
try:
    from yfinance.exceptions import YFPricesMissingError
except Exception:  # pragma: no cover - older yfinance
    from python.prefect.flows import YFPricesMissingError

from python.prefect.flows import _download_with_retry

def _has_net(host: str = "query1.finance.yahoo.com") -> bool:
    try:
        socket.create_connection((host, 443), timeout=5)
        return True
    except OSError:
        return False

pytestmark = pytest.mark.skipif(not _has_net(), reason="network required")

def test_minute_and_day_download():
    start = pd.Timestamp.utcnow() - pd.Timedelta(days=1)
    end = pd.Timestamp.utcnow()

    df_min = _download_with_retry("MSFT", start, end, "1m")
    assert df_min is not None
    assert not df_min.empty

    df_day = _download_with_retry("MSFT", start, end, "1d")
    assert df_day is not None
    assert not df_day.empty

def test_missing_error():
    with pytest.raises(YFPricesMissingError):
        yf.Ticker("INVALID$$").history(period="1d", interval="1d", raise_errors=True)

