from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor


def train(X: np.ndarray, y: np.ndarray, params: dict | None = None) -> dict:
    """Train a LightGBM regressor."""
    if params is None:
        params = {}
    model = LGBMRegressor(**params)
    model.fit(X, y)
    return {"model": model}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    """Predict using a trained LightGBM model."""
    return model["model"].predict(X)
