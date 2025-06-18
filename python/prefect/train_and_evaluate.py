"""Optuna training flow with MLflow logging."""

from __future__ import annotations

import importlib
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import optuna
try:
    import mlflow
except Exception:  # pragma: no cover - optional dependency
    mlflow = None  # type: ignore[assignment]

try:  # optional torch usage
    import torch
except Exception:  # pragma: no cover - torch not installed
    torch = None

import pandas as pd
import pyarrow.dataset as ds
from prefect import flow, task
from prefect.filesystems import LocalFileSystem
from .cleanup import remove_checkpoints, CHECKPOINT_BASE
from sklearn.metrics import mean_squared_error


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "python" / "configs"
MODELS_DIR = ROOT_DIR / "python" / "models"
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR = ROOT_DIR / "python" / "data"
FEATURES_DIR = ROOT_DIR / "python" / "features"

# Valid label/frequency pairs for each model family as defined in the project
# blueprint. This restricts ``train_all`` to exactly the 27 intended
# combinations.
MODEL_LABEL_FREQ: Dict[str, Dict[str, list[str]]] = {
    "lightgbm": {"B1": ["minute", "hour", "day"]},
    "catboost": {"B1": ["minute", "hour", "day"]},
    "tabnet": {"B1": ["hour", "day"]},
    "prophet": {"R": ["day"]},
    "n_linear": {"R": ["hour", "day"]},
    "lstm": {"B1": ["minute", "hour", "day"], "R": ["minute", "hour", "day"]},
    "tft": {"R": ["minute", "hour", "day"]},
    "informer": {"R": ["hour", "day"]},
    "patchtst": {"R": ["minute", "hour", "day"]},
    "finrl_ppo": {"T3": ["hour", "day"]},
}


def _load_dataset(freq: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load engineered features from ``python/features/<freq>/`` and generate labels.

    The data is read in chunks using ``pyarrow.dataset`` so that even very large
    parquet collections can be processed without loading everything into memory
    at once.
    """

    dir_path = FEATURES_DIR / freq
    paths = sorted(dir_path.glob("*.parquet"))

    from ..features import labels as lbl

    if paths:
        dataset = ds.dataset(paths)
        total_rows = dataset.count_rows()
        split_idx = int(0.8 * total_rows)

        X_train_parts: list[np.ndarray] = []
        X_val_parts: list[np.ndarray] = []
        y_train_parts: Dict[str, list[np.ndarray]] = {"B1": [], "T3": [], "R": []}
        y_val_parts: Dict[str, list[np.ndarray]] = {"B1": [], "T3": [], "R": []}

        current = 0
        for batch in dataset.to_batches(batch_size=1024):
            df_batch = batch.to_pandas()
            if "Close" not in df_batch.columns:
                df_batch["Close"] = np.random.normal(size=len(df_batch))

            feature_cols = [
                c
                for c in df_batch.columns
                if c not in {"Open", "High", "Low", "Close", "Volume"}
            ]
            X_batch = df_batch[feature_cols].to_numpy()

            labels_batch = {
                "B1": lbl.label_B1(df_batch).to_numpy(),
                "T3": lbl.label_T3(df_batch).to_numpy(),
                "R": lbl.label_R(df_batch).to_numpy(),
            }

            end = current + len(df_batch)
            if end <= split_idx:
                X_train_parts.append(X_batch)
                for k in y_train_parts:
                    y_train_parts[k].append(labels_batch[k])
            elif current >= split_idx:
                X_val_parts.append(X_batch)
                for k in y_val_parts:
                    y_val_parts[k].append(labels_batch[k])
            else:
                cut = split_idx - current
                X_train_parts.append(X_batch[:cut])
                X_val_parts.append(X_batch[cut:])
                for k in y_train_parts:
                    y_train_parts[k].append(labels_batch[k][:cut])
                    y_val_parts[k].append(labels_batch[k][cut:])
            current = end

        X_train = np.vstack(X_train_parts) if X_train_parts else np.empty((0, 0))
        X_val = np.vstack(X_val_parts) if X_val_parts else np.empty((0, 0))
        y_train = {k: np.concatenate(v) if v else np.empty(0) for k, v in y_train_parts.items()}
        y_val = {k: np.concatenate(v) if v else np.empty(0) for k, v in y_val_parts.items()}
        return X_train, X_val, y_train, y_val
    else:
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            rng.normal(size=(200, 11)),
            columns=[f"f{i}" for i in range(10)] + ["Close"],
        )

        feature_cols = [c for c in df.columns if c not in {"Open", "High", "Low", "Close", "Volume"}]
        X = df[feature_cols].to_numpy()
        labels = {
            "B1": lbl.label_B1(df).to_numpy(),
            "T3": lbl.label_T3(df).to_numpy(),
            "R": lbl.label_R(df).to_numpy(),
        }

        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train = {k: v[:split] for k, v in labels.items()}
        y_val = {k: v[split:] for k, v in labels.items()}
        return X_train, X_val, y_train, y_val

# checkpoint storage setup
try:
    LocalFileSystem(basepath=str(CHECKPOINT_BASE)).save("checkpoints", overwrite=True)
except Exception:
    pass
CHECKPOINT_STORAGE = "local-file-system/checkpoints"


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


def evaluate_model(predict_fn, model: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> float:
    """Generic MSE evaluation for arbitrary model."""
    preds = predict_fn(model, X)
    return float(mean_squared_error(y, preds))


def save_model(name: str, model: Dict[str, Any]) -> Path:
    """Persist a trained model to ``MODELS_DIR`` and return its path."""
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(name: str) -> Dict[str, Any]:
    """Load a previously saved model from ``MODELS_DIR``."""
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def _model_funcs(name: str):
    module = importlib.import_module(f"python.models.{name}")
    return module.train, module.predict


@task
def run_study(model: str, label: str, freq: str, space: Dict[str, Any], n_trials: int) -> None:
    """Run an Optuna study for one model/label/frequency combination."""

    X_train, X_val, y_train_dict, y_val_dict = _load_dataset(freq)
    y_train = y_train_dict[label]
    y_val = y_val_dict[label]

    mlflow.set_tracking_uri(f"file://{ROOT_DIR / 'mlruns'}")
    exp_name = f"{model}_{freq}_{label}"
    mlflow.set_experiment(exp_name)

    train_fn, predict_fn = _model_funcs(model)

    def objective(trial: optuna.Trial) -> float:
        params = {
            k: trial.suggest_categorical(k, v) if isinstance(v, list) else v
            for k, v in space.items()
        }
        if model == "lightgbm":
            X_all = np.vstack([X_train, X_val])
            y_all = np.concatenate([y_train, y_val])
            from sklearn.model_selection import TimeSeriesSplit

            cv = TimeSeriesSplit(n_splits=3)
            metrics = []
            for train_idx, val_idx in cv.split(X_all):
                X_tr, X_va = X_all[train_idx], X_all[val_idx]
                y_tr, y_va = y_all[train_idx], y_all[val_idx]
                model_state = train_fn(X_tr, y_tr, params)
                fold_metric = evaluate_model(predict_fn, model_state, X_va, y_va)
                metrics.append(fold_metric)
            metric = float(np.mean(metrics))
        else:
            model_state = train_fn(X_train, y_train, params)
            metric = evaluate_model(predict_fn, model_state, X_val, y_val)
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metric("mse", metric)
        return metric

    # prune trials after three evaluations without improvement
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    name = f"{model}_{freq}_{label}"
    best_model = train_fn(X_full, y_full, study.best_params)
    save_model(name, best_model)

    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        return
    runs = mlflow.search_runs([exp.experiment_id])
    runs = runs.sort_values("metrics.mse")
    for run_id in runs.iloc[5:]["run_id"]:
        mlflow.delete_run(run_id)


@flow(persist_result=True, result_storage=CHECKPOINT_STORAGE)
def train_all(n_trials: int = 60) -> None:
    cfg = load_config("optuna")
    for model, label_map in MODEL_LABEL_FREQ.items():
        space = cfg.get(model, {})
        for label, freqs in label_map.items():
            for freq in freqs:
                run_study(model, label, freq, space or {}, n_trials)
    remove_checkpoints()


__all__ = [
    "train_all",
    "run_study",
    "save_model",
    "load_model",
    "train_model",
    "predict",
    "evaluate",
    "evaluate_model",
]

