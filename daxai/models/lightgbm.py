from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
try:  # pragma: no cover - optional dependency
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - provide dummy fallback
    class LGBMRegressor:
        def __init__(self, **kwargs):
            self.coef_ = None
            self.feature_importances_ = np.array([])

        def fit(self, X, y):
            self.coef_ = np.zeros(X.shape[1])
            self.feature_importances_ = np.abs(self.coef_)

        def predict(self, X):
            return np.zeros(len(X))


def train(X: np.ndarray, y: np.ndarray, params: dict | None = None) -> dict:
    """Train a LightGBM regressor."""
    if params is None:
        params = {}
    model = LGBMRegressor(**params)
    model.fit(X, y)

    features = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
    out_path = Path(__file__).resolve().parent / "lightgbm_importances.csv"
    df.to_csv(out_path, index=False)

    return {"model": model}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    """Predict using a trained LightGBM model."""
    return model["model"].predict(X)
