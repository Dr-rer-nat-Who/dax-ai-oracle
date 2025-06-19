from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from daxai.prefect import backtest as bt


def test_backtest_produces_metrics(tmp_path: Path, monkeypatch) -> None:
    bt.MLRUNS_DIR = tmp_path / "mlruns"
    bt.BEST_DIR = bt.MLRUNS_DIR / "best"
    bt.DATA_DIR = tmp_path / "data"
    model_artifact = bt.MLRUNS_DIR / "0" / "run" / "artifacts"
    model_artifact.mkdir(parents=True)
    freq_dir = bt.DATA_DIR / "day"
    freq_dir.mkdir(parents=True)

    df = pd.DataFrame(
        {"Open": [1, 2], "High": [1, 2], "Low": [1, 2], "Close": [1, 2]},
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
    )
    df.to_parquet(freq_dir / "sample.parquet")

    model = {"weights": [1.0] * 10}
    with open(model_artifact / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    monkeypatch.setattr(bt, "_vectorbt_metrics", lambda p, pr: (0.5, 0.5, 0.5, pr))
    monkeypatch.setattr(
        bt,
        "_backtrader_metrics",
        lambda p, s: (0.1, 0.1, 0.1, 0.1, pd.Series([1, 2], index=p.index)),
    )

    results = bt.backtest.fn(
        mlruns_dir=bt.MLRUNS_DIR, best_dir=bt.BEST_DIR, data_dir=bt.DATA_DIR
    )
    assert results, "No results returned"
    metrics_path = bt.BEST_DIR / "metrics.csv"
    assert metrics_path.exists(), "metrics.csv not created"
    df = pd.read_csv(metrics_path)
    assert {"accuracy", "balanced_accuracy", "f1", "sharpe", "final_score"}.issubset(df.columns)


def test_make_features_and_predict():
    prices = bt._make_price_data(5)
    feats = bt._make_features(prices, 3)
    assert feats.shape == (5, 3)
    model = {"weights": [1.0, 2.0, 3.0]}
    preds = bt._predict(model, feats)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 5


def test_backtest_triggers_gc(tmp_path: Path, monkeypatch) -> None:
    bt.MLRUNS_DIR = tmp_path / "mlruns"
    bt.BEST_DIR = bt.MLRUNS_DIR / "best"
    bt.DATA_DIR = tmp_path / "data"
    model_artifact = bt.MLRUNS_DIR / "0" / "run" / "artifacts"
    model_artifact.mkdir(parents=True)
    freq_dir = bt.DATA_DIR / "day"
    freq_dir.mkdir(parents=True)

    df = pd.DataFrame(
        {"Open": [1], "High": [1], "Low": [1], "Close": [1]},
        index=pd.date_range("2020-01-01", periods=1, freq="D"),
    )
    df.to_parquet(freq_dir / "sample.parquet")

    with open(model_artifact / "model.pkl", "wb") as f:
        pickle.dump({"weights": [1.0]}, f)

    monkeypatch.setattr(bt, "_vectorbt_metrics", lambda p, pr: (0.5, 0.5, 0.5, pr))
    monkeypatch.setattr(
        bt,
        "_backtrader_metrics",
        lambda p, s: (0.1, 0.1, 0.1, 0.1, pd.Series([1], index=p.index)),
    )

    called = {"gc": False}

    def fake_gc():
        called["gc"] = True

    monkeypatch.setattr(bt, "dvc_gc_workspace", fake_gc)

    bt.backtest.fn(mlruns_dir=bt.MLRUNS_DIR, best_dir=bt.BEST_DIR, data_dir=bt.DATA_DIR)

    assert called["gc"], "dvc gc not triggered"
