"""TabNet regressor using ``pytorch_tabnet``."""

from __future__ import annotations

import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor


def train(X: np.ndarray, y: np.ndarray, params: dict | None = None) -> dict:
    """Train a TabNet regressor."""
    if params is None:
        params = {}

    epochs = int(params.pop("epochs", 10))
    model = TabNetRegressor(**params)
    model.fit(X, y, max_epochs=epochs)
    return {"model": model}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    """Generate predictions with a trained TabNet model."""
    return model["model"].predict(X).squeeze()


