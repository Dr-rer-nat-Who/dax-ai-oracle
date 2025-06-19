import importlib
import sys
import types
import builtins
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))


def load_app(yf_module):
    sys.modules.pop('python.dashboard.app', None)
    if yf_module is None:
        sys.modules.pop('yfinance', None)
        import builtins
        orig_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == 'yfinance':
                raise ImportError
            return orig_import(name, *args, **kwargs)

        builtins.__import__, token = fake_import, orig_import
        try:
            return importlib.import_module('python.dashboard.app')
        finally:
            builtins.__import__ = token
    else:
        sys.modules['yfinance'] = yf_module
        return importlib.import_module('python.dashboard.app')


def test_caption_with_yfinance(monkeypatch):
    msgs = []
    monkeypatch.setattr(st.sidebar, 'caption', lambda msg: msgs.append(msg))
    yf_mod = types.SimpleNamespace(__version__='0.1', download=lambda *a, **k: None)
    load_app(yf_mod)
    assert msgs[-1] == 'yfinance 0.1'


def test_caption_without_yfinance(monkeypatch):
    msgs = []
    monkeypatch.setattr(st.sidebar, 'caption', lambda msg: msgs.append(msg))
    load_app(None)
    assert msgs[-1] == 'yfinance unavailable'


import pandas as pd
import streamlit as st



def reload_app(monkeypatch, import_error=False):
    """Reload dashboard.app with optional ImportError for plotly."""
    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if import_error and name == "plotly.express":
            raise ImportError
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    if "python.dashboard.app" in importlib.sys.modules:
        del importlib.sys.modules["python.dashboard.app"]
    return importlib.import_module("python.dashboard.app")


def setup_equity(tmp_path, app):
    best = tmp_path / "mlruns" / "best" / "model1"
    best.mkdir(parents=True)
    series = pd.Series([1, 2, 3])
    series.to_csv(best / "test_equity.csv")
    app.BEST_DIR = tmp_path / "mlruns" / "best"


def test_plotly_missing(monkeypatch, tmp_path):
    warnings = []
    monkeypatch.setattr(st, "warning", lambda msg: warnings.append(msg))
    line_calls = []
    monkeypatch.setattr(st, "line_chart", lambda data: line_calls.append(data))
    monkeypatch.setattr(st, "plotly_chart", lambda fig: (_ for _ in ()).throw(AssertionError("plotly_chart called")))

    app = reload_app(monkeypatch, import_error=True)
    setup_equity(tmp_path, app)

    app.show_equity()

    assert "Plotly not installed" in warnings[0]
    assert line_calls, "line_chart not called"


def test_plotly_available(monkeypatch, tmp_path):
    warnings = []
    monkeypatch.setattr(st, "warning", lambda msg: warnings.append(msg))
    plot_calls = []
    def fake_plotly(fig):
        plot_calls.append(fig)
    monkeypatch.setattr(st, "plotly_chart", fake_plotly)
    line_calls = []
    monkeypatch.setattr(st, "line_chart", lambda data: line_calls.append(data))

    app = reload_app(monkeypatch, import_error=False)
    setup_equity(tmp_path, app)

    # patch px.line to return a dummy figure
    class DummyFig:
        pass
    monkeypatch.setattr(app.px, "line", lambda *a, **k: DummyFig())

    app.show_equity()

    assert not warnings, "warning emitted"
    assert plot_calls, "plotly_chart not called"
    assert not line_calls, "line_chart should not be used when plotly available"
