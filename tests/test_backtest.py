from pathlib import Path
import pickle
import sys
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from python.prefect import backtest as bt


def test_backtest_produces_metrics(tmp_path: Path) -> None:
    bt.MODELS_DIR = tmp_path / "models"
    bt.BEST_DIR = tmp_path / "best"
    bt.MODELS_DIR.mkdir()
    model = {"weights": [1.0] * 10}
    with open(bt.MODELS_DIR / "dummy.pkl", "wb") as f:
        pickle.dump(model, f)

    results = bt.backtest.fn(models_dir=bt.MODELS_DIR, best_dir=bt.BEST_DIR)
    assert results, "No results returned"
    metrics_path = bt.BEST_DIR / "metrics.csv"
    assert metrics_path.exists(), "metrics.csv not created"
    df = pd.read_csv(metrics_path)
    assert {"accuracy", "balanced_accuracy", "f1", "sharpe", "final_score"}.issubset(df.columns)
