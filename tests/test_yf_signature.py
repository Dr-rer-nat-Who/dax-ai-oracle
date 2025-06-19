import sys
import importlib
import inspect
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_download_signature(monkeypatch):
    if isinstance(sys.modules.get("yfinance"), types.SimpleNamespace):
        del sys.modules["yfinance"]
    yf = importlib.import_module("yfinance")

    class DummyLogger:
        def info(self, *a, **k):
            pass
        def warning(self, *a, **k):
            pass
        def error(self, *a, **k):
            pass

    class DummyLocalFileSystem:
        def __init__(self, basepath: str):
            pass
        def save(self, *a, **k):
            pass

    def dummy_dec(*a, **k):
        def wrapper(fn):
            return fn
        return wrapper

    class DummyFlowRunContext:
        @staticmethod
        def get():
            return None

    prefect_stub = types.SimpleNamespace(
        flow=dummy_dec,
        task=dummy_dec,
        get_run_logger=lambda: DummyLogger(),
        filesystems=types.SimpleNamespace(LocalFileSystem=DummyLocalFileSystem),
        runtime=types.SimpleNamespace(
            flow_run=types.SimpleNamespace(FlowRunContext=DummyFlowRunContext)
        ),
        task_runners=types.SimpleNamespace(SequentialTaskRunner=object),
    )

    monkeypatch.setitem(sys.modules, "prefect", prefect_stub)
    monkeypatch.setitem(sys.modules, "prefect.filesystems", prefect_stub.filesystems)
    monkeypatch.setitem(sys.modules, "prefect.runtime", prefect_stub.runtime)
    monkeypatch.setitem(
        sys.modules,
        "prefect.runtime.flow_run",
        prefect_stub.runtime.flow_run,
    )
    monkeypatch.setitem(
        sys.modules,
        "prefect.task_runners",
        prefect_stub.task_runners,
    )
    if "python.prefect.flows" in sys.modules:
        del sys.modules["python.prefect.flows"]
    flows = importlib.import_module("python.prefect.flows")

    params = set(inspect.signature(yf.download).parameters)
    passed = {
        "start",
        "end",
        "interval",
        "auto_adjust",
        "progress",
    } | set(flows._COMPAT_ARGS.keys())

    missing = passed - params
    assert not missing, f"Removed args passed to yf.download: {sorted(missing)}"
