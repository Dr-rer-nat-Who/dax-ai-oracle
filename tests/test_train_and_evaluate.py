from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from python.prefect.train_and_evaluate import run_study, load_config


class DummyMlflow:
    def set_tracking_uri(self, uri):
        pass

    def set_experiment(self, name):
        pass

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    def start_run(self):
        return self.DummyRun()

    def log_params(self, params):
        pass

    def log_metric(self, name, value):
        pass

    def get_experiment_by_name(self, name):
        class E:
            experiment_id = "0"
        return E()

    def search_runs(self, ids):
        import pandas as pd

        return pd.DataFrame([], columns=["metrics.mse", "run_id"])

    def delete_run(self, run_id):
        pass


def test_lightgbm_model_trains(tmp_path: Path, monkeypatch) -> None:
    cfg = load_config("optuna")
    monkeypatch.setattr("python.prefect.train_and_evaluate.mlflow", DummyMlflow())
    from python.prefect import train_and_evaluate as te

    te.DATA_DIR = tmp_path
    freq_dir = tmp_path / "day"
    freq_dir.mkdir()
    df = pd.DataFrame(
        {
            "Open": np.linspace(1, 2, 20),
            "High": np.linspace(1, 2, 20),
            "Low": np.linspace(1, 2, 20),
            "Close": np.linspace(1, 2, 20),
            "f0": np.random.rand(20),
        }
    )
    df.to_parquet(freq_dir / "sample.parquet")

    run_study.fn("lightgbm", "B1", "day", cfg["lightgbm"], n_trials=1)


def test_train_model_predict_evaluate(monkeypatch):
    from python.prefect import train_and_evaluate as te

    monkeypatch.setattr(te, "torch", None)
    X = np.random.rand(20, 2)
    y = X.dot(np.array([1.5, -2.0]))
    model = te.train_model({"lr": 0.1, "epochs": 5}, X, y)
    preds = te.predict(model, X)
    assert preds.shape == (20,)
    mse = te.evaluate(model, X, y)
    assert mse >= 0


def test_range_sampling(monkeypatch):
    from python.prefect import train_and_evaluate as te

    monkeypatch.setattr(te, "mlflow", DummyMlflow())

    captured: list[dict] = []

    def dummy_train(X, y, params):
        captured.append(params)
        return {}

    def dummy_predict(model, X):
        return np.zeros(len(X))

    monkeypatch.setattr(te, "_model_funcs", lambda name: (dummy_train, dummy_predict))

    class DummyTrial:
        def __init__(self):
            self.params = {}

        def suggest_int(self, name, low, high):
            val = (low + high) // 2
            self.params[name] = val
            return val

        def suggest_float(self, name, low, high):
            val = (low + high) / 2.0
            self.params[name] = val
            return val

        def suggest_categorical(self, name, choices):
            val = choices[0]
            self.params[name] = val
            return val

    class DummyStudy:
        def __init__(self):
            self.trial = DummyTrial()

        def optimize(self, func, n_trials):
            func(self.trial)

        @property
        def best_params(self):
            return self.trial.params

    monkeypatch.setattr(te.optuna, "create_study", lambda direction=None, pruner=None: DummyStudy())

    monkeypatch.setattr(te, "_load_dataset", lambda freq: (np.zeros((2, 2)), np.zeros((2, 2)), {"B1": np.zeros(2)}, {"B1": np.zeros(2)}))

    space = {"a": [1, 3], "b": [0.0, 1.0]}

    te.run_study.fn("dummy", "B1", "day", space, n_trials=1)

    assert captured
    params = captured[0]
    assert params["a"] == 2
    assert abs(params["b"] - 0.5) < 1e-6
