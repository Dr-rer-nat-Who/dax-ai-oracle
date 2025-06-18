from __future__ import annotations

import numpy as np
from catboost import CatBoostRegressor


def train(X: np.ndarray, y: np.ndarray, params: dict | None = None) -> dict:
    """Train a CatBoost regressor."""
    if params is None:
        params = {}
    model = CatBoostRegressor(**params, verbose=False)
    model.fit(X, y)
    return {"model": model}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    """Predict using a trained CatBoost model."""
    return model["model"].predict(X)
