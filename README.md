# DAX AI Oracle

This repository contains a local workflow for ingesting DAX market data, training multiple model families and exposing results through a Streamlit dashboard.

## Architecture overview

```
Prefect Flows -> Data Lake (Parquet + DVC) -> Feature Pipelines
     |                                        |
     |                                        v
     +------> Optuna Studies ----> MLflow Runs ----> Backtesting
                                              |
                                              v
                                        Streamlit Dashboard
```
- **Task orchestration**: Prefect 2.0 with checkpoints.
- **Artifact versioning**: MLflow for models and DVC for large datasets.
- **Backtesting**: Vectorbt for quick screening and Backtrader for realistic simulation.

### Data locations

| Frequency | History       | Target path        | Rotation |
|-----------|---------------|--------------------|-----------------------------|
| Minute    | last 90 days  | `data/raw/1min/`   | older blocks aggregated to 5‑min on cleanup |
| Hour      | 24 months     | `data/raw/1h/`     | kept |
| Day       | 10 years      | `data/raw/1d/`     | kept |

Data is pulled from `yfinance` and stored as Parquet with DVC deduplication.

### Model families

```
LightGBM   | CatBoost   | TabNet      | Prophet    | N-Linear
LSTM       | TFT        | Autoformer  | Informer   | PatchTST
TimesNet   | FinRL-PPO
```
All combinations across three frequencies and three label types yield 27 model variants.

More details are available in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## CLI usage

Run the full pipeline and start the dashboard with:

```bash
python cli.py run-all --freq all --cleanup yes
```

This command ingests data, trains models, backtests, cleans up and launches the dashboard at `localhost:8501`.

## Streamlit dashboard

After running the flows you can also launch the dashboard manually:

```bash
streamlit run python/dashboard/app.py
```

## Development

All documentation resides in the [docs](docs/) folder.
Run the tests with:

```bash
pytest -q
```

## Local setup on macOS (Apple Silicon)

The project has been tested on a MacBook M1. Follow these steps to set up the
environment:

1. Install the command line tools and Homebrew:

   ```bash
   xcode-select --install
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Install required system packages via Homebrew. `ta-lib` is needed for the
   feature pipeline and `dvc`/`mlflow` provide data and model tracking:

   ```bash
   brew install python@3.12 node ta-lib dvc mlflow
   ```

3. Create a Python virtual environment and activate it:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install PyTorch with MPS support and the remaining Python requirements:

   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/mps
   pip install -r requirements.txt
   ```

5. Install the Node dependencies for the dashboard (uses Vite/React):

   ```bash
   npm install
   ```

After these steps the command below will ingest data, train the models,
backtest and launch the Streamlit dashboard on `localhost:8501`:

```bash
python cli.py run-all --freq all --cleanup yes
```

