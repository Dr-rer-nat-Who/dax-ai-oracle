import importlib
import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st


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
