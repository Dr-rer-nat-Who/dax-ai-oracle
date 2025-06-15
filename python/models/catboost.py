from __future__ import annotations

import numpy as np
from .common import gd_train, predict as _predict


def train(X: np.ndarray, y: np.ndarray, params: dict | None = None) -> dict:
    if params is None:
        params = {}
    lr = float(params.get("lr", 0.01))
    epochs = int(params.get("epochs", 10))
    weights = gd_train(X, y, lr, epochs)
    return {"weights": weights}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    return _predict(model["weights"], X)
