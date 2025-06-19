import importlib
import builtins
import sys
import pytest


def test_prophet_missing_raises(monkeypatch):
    if 'python.models.prophet' in sys.modules:
        del sys.modules['python.models.prophet']

    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'prophet':
            raise ImportError('no prophet')
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', fake_import)

    with pytest.raises(RuntimeError) as exc:
        importlib.import_module('python.models.prophet')

    assert "Prophet-Modell" in str(exc.value)
