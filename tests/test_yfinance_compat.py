import importlib
import types
import sys
import pandas as pd


def load_flows(has_threads: bool, monkeypatch):
    if has_threads:
        def download(ticker, start=None, end=None, interval=None, auto_adjust=False, progress=True, threads=True):
            return pd.DataFrame()
    else:
        def download(ticker, start=None, end=None, interval=None, auto_adjust=False, progress=True):
            return pd.DataFrame()
    stub = types.SimpleNamespace(download=download)
    monkeypatch.setitem(sys.modules, 'yfinance', stub)
    if 'python.prefect.flows' in sys.modules:
        del sys.modules['python.prefect.flows']
    return importlib.import_module('python.prefect.flows')


def load_app(has_threads: bool, monkeypatch):
    if has_threads:
        def download(ticker, period=None, interval=None, progress=True, threads=True):
            return pd.DataFrame({'Close': [1]})
    else:
        def download(ticker, period=None, interval=None, progress=True):
            return pd.DataFrame({'Close': [1]})
    stub = types.SimpleNamespace(download=download)
    monkeypatch.setitem(sys.modules, 'yfinance', stub)
    dummy_st = types.SimpleNamespace(
        header=lambda *a, **k: None,
        line_chart=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        info=lambda *a, **k: None,
        experimental_rerun=lambda: None,
        session_state={},
    )
    monkeypatch.setitem(sys.modules, 'streamlit', dummy_st)
    if 'python.dashboard.app' in sys.modules:
        del sys.modules['python.dashboard.app']
    return importlib.import_module('python.dashboard.app')


def test_flows_with_threads(monkeypatch):
    flows = load_flows(True, monkeypatch)
    called = {}
    def record(*a, **k):
        called.update(k)
        return pd.DataFrame()
    monkeypatch.setattr(flows, 'yf', types.SimpleNamespace(download=record))
    flows._download_with_retry('T', pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02'), '1d', attempts=1)
    assert flows._COMPAT_ARGS == {'threads': False}
    assert called.get('threads') is False
    assert 'progress' not in called


def test_flows_without_threads(monkeypatch):
    flows = load_flows(False, monkeypatch)
    called = {}
    def record(*a, **k):
        called.update(k)
        return pd.DataFrame()
    monkeypatch.setattr(flows, 'yf', types.SimpleNamespace(download=record))
    flows._download_with_retry('T', pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02'), '1d', attempts=1)
    assert flows._COMPAT_ARGS == {}
    assert 'threads' not in called
    assert 'progress' not in called


def test_app_with_threads(monkeypatch):
    app = load_app(True, monkeypatch)
    assert app._COMPAT_ARGS == {'threads': False}


def test_app_without_threads(monkeypatch):
    app = load_app(False, monkeypatch)
    assert app._COMPAT_ARGS == {}
