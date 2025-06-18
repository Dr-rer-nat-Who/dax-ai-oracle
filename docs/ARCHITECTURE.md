# Project Architecture

This document provides an end-to-end summary of the local DAX analytics system. The pipeline ingests price data, builds features, trains multiple model families, backtests the results and exposes everything through a Streamlit dashboard.

## 1. Overview

```text
Prefect Flows -> Data Lake (Parquet + DVC) -> Feature Pipelines
     |                                        |
     |                                        v
     +------> Optuna Studies ----> MLflow Runs ----> Backtesting
                                              |
                                              v
                                        Streamlit Dashboard
```

- **Task orchestration**: Prefect 2.0 with checkpoints to resume interrupted runs.
- **Artifact versioning**: MLflow for models, DVC for heavy datasets.
- **Backtesting**: Vectorbt for quick screening and Backtrader for realistic broker simulation.

## 2. Data Layer

| Frequency | History       | Target path        | Rotation                                    |
|-----------|---------------|--------------------|---------------------------------------------|
| Minute    | last 90 days  | `data/raw/1min/`   | older blocks aggregated to 5‑min on cleanup |
| Hour      | 24 months     | `data/raw/1h/`     | kept                                         |
| Day       | 10 years      | `data/raw/1d/`     | kept                                         |

Data comes from `yfinance` and is stored as Parquet with DVC for deduplicated storage.

## 3. Feature Engineering

1. Technical indicators via TA‑Lib (≈245 features).
2. Price‑action statistics such as wick ratio and ATR.
3. Window embeddings of normalized returns for the last 32 or 64 steps.
4. Datetime features (weekday, month end, holiday flag).
5. Optional exogenous inputs like DAX future spread.

Transformer models operate on raw window tensors, while tabular models use flattened features.

## 4. Label Schemes

| Name | Definition                                   | Use case |
|------|----------------------------------------------|----------|
| B1   | Binary ↑/↓, horizon = 1 step                 | classic ML |
| T3   | +1 > +0.2%, 0 ∈ ±0.2%, −1 < −0.2%            | RL, multi‑class |
| R    | Log‑return in % for one step                 | regressors |

## 5. Model Portfolio

```text
LightGBM   | CatBoost   | TabNet      | Prophet    | N-Linear
LSTM       | TFT        | Autoformer  | Informer   | PatchTST
TimesNet   | FinRL-PPO
```

All combinations across 3 frequencies and 3 label types yield 27 model variants. Each Optuna study runs about 60 trials with early stopping via a median pruner (three warmup steps).

## 6. Training Pipeline

1. `train_and_evaluate` Prefect flow pulls batches from Parquet, launches one Optuna study per model and logs trials to MLflow.
2. Checkpoints reside in `~/checkpoints/{flow_run}` and are removed after success.
3. A cleanup flow runs `dvc gc -w` after each backtest and drops Parquet partitions no longer referenced by the Prefect cache.

## 7. Backtesting & Evaluation

- **Vectorbt** simulates step orders with 0.01 % slippage and reports accuracy metrics.
- **Backtrader** adds 0.04 % fees and 0.05 % slippage for realistic results (Sharpe, Sortino, drawdown).
- Ranking score: `0.4 * balanced_accuracy + 0.6 * min(Sharpe / 3, 1)`.
- Only models scoring ≥ 0.6 are kept in `mlruns/best/`.

## 8. Frontend

Streamlit (auto refresh every 60 s) exposes live signals, a leaderboard and performance plots. It loads only the pickled models and metrics CSVs.

## 9. Resources & Runtime

- Apple M1 hardware; PyTorch uses Metal‑MPS for ~7–8× speed‑up.
- Typical runtime: tabular minute models ~30 min per study, PatchTST ~4 h. The entire set of 27 models completes in roughly 18–22 h.

## 10. Directory Layout

```text
cli.py
configs/
    data.yaml
    optuna.yaml
    cleanup.yaml
data/
features/
models/
prefect/
    ingest.py
    feature_build.py
    train_and_evaluate.py
    backtest.py
    cleanup.py
dashboard/
```

Run everything via

```bash
python cli.py run-all --freq all --cleanup yes
```

which ingests data, trains models, backtests, cleans up and launches the dashboard at `localhost:8501`.

## 11. Operation & Monitoring

- Prefect UI (port 4200) shows logs and resumes flows after crashes.
- MLflow UI (port 5000) compares metrics and allows artifact download.
- A disk guard triggers cleanup when free space drops below 5 GB.

## 12. Optional Extensions

- Online learning with `river` for minute data.
- Ensemble stacking (e.g. LightGBM + PatchTST output into a Meta‑CatBoost).
- Order routing to IBKR TWS through the Backtrader‑IB API.

