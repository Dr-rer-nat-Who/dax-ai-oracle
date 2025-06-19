from pathlib import Path

import python.cli as cli


def test_run_all_invokes_flows(monkeypatch):
    calls = []
    def record(name):
        def inner(*args, **kwargs):
            calls.append(name)
        return inner
    monkeypatch.setattr(cli, "init_prefect", lambda: None)
    monkeypatch.setattr(cli, "ingest", record("ingest"))
    monkeypatch.setattr(cli, "feature_build", record("feature_build"))
    monkeypatch.setattr(cli, "train_all", record("train_all"))
    monkeypatch.setattr(cli, "backtest", record("backtest"))
    monkeypatch.setattr(cli, "cleanup", record("cleanup"))
    monkeypatch.setattr(cli, "_disk_free_gb", lambda path=Path("."): 10)

    spawned = {}
    class DummyPopen:
        def __init__(self, args, *a, **k):
            spawned["args"] = args
    monkeypatch.setattr(cli.subprocess, "Popen", DummyPopen)

    cli.main(["run-all", "--freq", "day", "--cleanup", "no"])

    assert calls == ["ingest", "feature_build", "train_all", "backtest"]
    assert spawned.get("args", [])[0] == "streamlit"
