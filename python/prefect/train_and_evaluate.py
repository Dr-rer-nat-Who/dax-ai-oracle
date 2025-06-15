"""Optuna training flow with MLflow logging."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import optuna
import mlflow

try:  # optional torch usage
    import torch
except Exception:  # pragma: no cover - torch not installed
    torch = None

import pandas as pd
from prefect import flow, task


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "python" / "configs"
MODELS_DIR = ROOT_DIR / "python" / "models"
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR = ROOT_DIR / "python" / "data"


def _load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load feature and label arrays with a train/val split."""
    feats = DATA_DIR / "features.parquet"
    labels = DATA_DIR / "labels.parquet"
    if feats.exists() and labels.exists():
        X_df = pd.read_parquet(feats)
        y_df = pd.read_parquet(labels)
        X = X_df.to_numpy()
        y = y_df.iloc[:, 0].to_numpy()
    else:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(200, 10))
        true_w = rng.normal(size=10)
        y = X.dot(true_w) + rng.normal(scale=0.1, size=200)
    split = int(0.8 * len(X))
    return X[:split], X[split:], y[:split], y[split:]


def load_config(name: str) -> Dict[str, Any]:
    with open(CONFIG_DIR / f"{name}.yaml", "r") as f:
        import yaml

        return yaml.safe_load(f)


def _device() -> str:
    if torch is None:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _train_numpy(X: np.ndarray, y: np.ndarray, lr: float, epochs: int) -> np.ndarray:
    """Simple gradient descent linear regression using numpy."""
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        preds = X.dot(w)
        grad = X.T.dot(preds - y) / len(y)
        w -= lr * grad
    return w


def _train_torch(X: np.ndarray, y: np.ndarray, lr: float, epochs: int) -> np.ndarray:
    """Linear regression using torch for optional GPU/MPS."""
    device = _device()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    w = torch.zeros(X.shape[1], device=device, requires_grad=True)
    opt = torch.optim.SGD([w], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.mean((X_t.mv(w) - y_t) ** 2)
        loss.backward()
        opt.step()
    return w.detach().cpu().numpy()


def train_model(params: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    lr = float(params.get("lr", 0.01))
    epochs = int(params.get("epochs", 10))
    if torch is not None:
        weights = _train_torch(X, y, lr, epochs)
    else:
        weights = _train_numpy(X, y, lr, epochs)
    return {"weights": weights}


def predict(model: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    return X.dot(model["weights"])


def evaluate(model: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> float:
    preds = predict(model, X)
    return float(np.mean((y - preds) ** 2))


@task
def run_study(name: str, space: Dict[str, Any], n_trials: int) -> None:
    """Run an Optuna study for a single model family."""

    X_train, X_val, y_train, y_val = _load_dataset()

    mlflow.set_tracking_uri(f"file://{ROOT_DIR / 'mlruns'}")
    mlflow.set_experiment(name)

    def objective(trial: optuna.Trial) -> float:
        params = {
            k: trial.suggest_categorical(k, v) if isinstance(v, list) else v
            for k, v in space.items()
        }
        model = train_model(params, X_train, y_train)
        metric = evaluate(model, X_val, y_val)
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metric("mse", metric)
        return metric

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    best_model = train_model(study.best_params, X_full, y_full)
    with open(MODELS_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(best_model, f)

    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return
    runs = mlflow.search_runs([exp.experiment_id])
    runs = runs.sort_values("metrics.mse")
    for run_id in runs.iloc[5:]["run_id"]:
        mlflow.delete_run(run_id)


@flow
def train_all(n_trials: int = 60) -> None:
    cfg = load_config("optuna")
    for name, space in cfg.items():
        run_study(name, space or {}, n_trials)


__all__ = ["train_all", "run_study"]

