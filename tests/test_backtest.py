from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from python.prefect import backtest as bt


def test_backtest_produces_metrics(tmp_path: Path, monkeypatch) -> None:
    bt.MODELS_DIR = tmp_path / "models"
    bt.BEST_DIR = tmp_path / "best"
    bt.DATA_DIR = tmp_path / "data"
    bt.MODELS_DIR.mkdir()
    bt.DATA_DIR.mkdir()

    df = pd.DataFrame(
        {"Open": [1, 2], "High": [1, 2], "Low": [1, 2], "Close": [1, 2]},
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
    )
    df.to_parquet(bt.DATA_DIR / "sample.parquet")

    model = {"weights": [1.0] * 10}
    with open(bt.MODELS_DIR / "dummy.pkl", "wb") as f:
        pickle.dump(model, f)

    monkeypatch.setattr(bt, "_vectorbt_metrics", lambda p, pr: (0.5, 0.5, 0.5, pr))
    monkeypatch.setattr(
        bt,
        "_backtrader_metrics",
        lambda p, s: (0.1, 0.1, 0.1, 0.1, pd.Series([1, 2], index=p.index)),
    )

    results = bt.backtest.fn(
        models_dir=bt.MODELS_DIR, best_dir=bt.BEST_DIR, data_dir=bt.DATA_DIR
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
