"""Prophet time series forecasting model."""

from __future__ import annotations

import numpy as np
import pandas as pd
try:
    from prophet import Prophet
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "Prophet-Modell angefordert, 'prophet' fehlt. Installiere mit extras 'experiments'."
    ) from exc


def train(X: np.ndarray, y: np.ndarray, params: dict | None = None) -> dict:
    """Train a Prophet model.

    ``X`` is ignored as Prophet only needs the target series.
    """

    if params is None:
        params = {}

    df = pd.DataFrame({
        "ds": pd.date_range("2000-01-01", periods=len(y), freq="D"),
        "y": y,
    })
    model = Prophet(**params)
    model.fit(df)
    return {"model": model}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    """Predict using the trained Prophet model."""

    future = pd.DataFrame({
        "ds": pd.date_range("2000-01-01", periods=len(X), freq="D")
    })
    forecast = model["model"].predict(future)
    return forecast["yhat"].to_numpy()


