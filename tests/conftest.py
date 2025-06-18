import types, sys
if 'yfinance' not in sys.modules:
    yf = types.SimpleNamespace(download=lambda *a, **k: None)
    sys.modules['yfinance'] = yf
