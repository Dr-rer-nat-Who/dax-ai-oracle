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
Minute downloads are requested in eight-day windows to stay within the API limits.
If a start date lies more than 30 days in the past, it is adjusted to this
cutoff because minute history is only available for the recent month.
Tickers that fail to download are skipped with a warning.

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

All documentation resides in the [docs](docs/) folder. The minimum supported
Python version is **3.11**.
Run the tests with:

```bash
pytest -q
```

## Local setup on macOS (Apple Silicon)

The project has been tested on a MacBook M1. Follow these steps to set up the
environment:

0. Clone this repository (DVC requires a Git repository):

   ```bash
   git clone https://github.com/yourname/dax-ai-oracle.git
   cd dax-ai-oracle
   ```

1. Install the command line tools and Homebrew:

   ```bash
   xcode-select --install
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Install required system packages via Homebrew. `ta-lib` and `dvc` are needed
   for feature generation and data versioning. `mlflow` will be installed later
   via `pip`:

   ```bash
   brew install python@3.11 node ta-lib dvc
   ```
   On Linux you can build TA‑Lib manually:

   ```bash
   wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
   tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib
   ./configure --prefix=/usr/local && make && sudo make install
   ```

3. Create a Python virtual environment and activate it:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

4. Install PyTorch and the remaining Python requirements. For CPU only:

   ```bash
   pip install torch torchvision torchaudio
   ```
   For NVIDIA GPUs use the CUDA 12.1 wheels:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   macOS users with Apple Silicon can instead enable MPS:

   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/mps
   ```

   Afterwards install the remaining requirements:

   ```bash
   pip install -r requirements.txt
   ```

5. Install the Node dependencies for the dashboard (uses Vite/React):

   ```bash
   npm install
   ```

6. (Optional) Install additional Python extras:

   ```bash
   pip install .[dashboard]
   pip install .[experiments]
   ```

7. Disable the default yfinance SQLite cache (or set a writable cache path) to
   avoid `OperationalError: unable to open database file` errors:

   ```bash
   export YFINANCE_NO_CACHE=1
   ```

After these steps the command below will ingest data, train the models,
backtest and launch the Streamlit dashboard on `localhost:8501`:

```bash
python cli.py run-all --freq all --cleanup yes
```

