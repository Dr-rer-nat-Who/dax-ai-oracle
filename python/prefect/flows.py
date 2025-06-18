from pathlib import Path
import subprocess
import sys

import pandas as pd
import yaml
import yfinance as yf
from prefect import flow, task


sys.path.append(str(Path(__file__).resolve().parents[1]))
from features.pipelines import compute_features

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "python" / "data"
FEATURES_DIR = ROOT_DIR / "python" / "features"

CHECKPOINT_BASE = Path.home() / "checkpoints"
# ensure the result storage block exists for local checkpoints
try:
    LocalFileSystem(basepath=str(CHECKPOINT_BASE)).save("checkpoints", overwrite=True)
except Exception:
    pass
CHECKPOINT_STORAGE = "local-file-system/checkpoints"


def load_config(name: str) -> dict:
    """Load a YAML config by name from the configs directory."""
    with open(CONFIG_DIR / f"{name}.yaml", "r") as f:
        return yaml.safe_load(f)


@task(log_prints=True)
def ensure_dvc_repo() -> None:
    """Initialize DVC in the repository if not already initialized."""
    if not (ROOT_DIR / ".dvc").exists():
        subprocess.run(["dvc", "init", "-q"], cwd=ROOT_DIR, check=True)


@task(log_prints=True)
def fetch_and_store(ticker: str, start: str, end: str, freq: str) -> Path:
    """Fetch OHLCV data from yfinance and store it as Parquet with DVC."""
    interval_map = {"minute": "1m", "hour": "1h", "day": "1d"}
    interval = interval_map[freq]
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    dest_dir = DATA_DIR / freq
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{ticker}.parquet"
    df.to_parquet(path)
    subprocess.run(["dvc", "add", str(path)], cwd=ROOT_DIR, check=True)
    return path


@task(log_prints=True)
def ensure_disk_space(threshold_gb: float = 5.0) -> None:
    """Trigger cleanup when free disk space drops below ``threshold_gb``."""
    free_gb = _disk_free_gb(ROOT_DIR)
    if free_gb < threshold_gb:
        print(f"Low disk space ({free_gb:.2f} GB); running cleanup")
        cleanup_flow()


@task(log_prints=True)
def build_features(path: Path, exogenous: Path | None = None) -> Path:
    """Read OHLCV parquet, generate features and store them with DVC."""
    df = pd.read_parquet(path)
    exo_df = pd.read_parquet(exogenous) if exogenous else None
    feats = compute_features(df, exo_df)
    dest_dir = FEATURES_DIR / path.parent.name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    feats.to_parquet(dest)
    subprocess.run(["dvc", "add", str(dest)], cwd=ROOT_DIR, check=True)
    return dest

@flow(persist_result=True, result_storage=CHECKPOINT_STORAGE)
def ingest(freq: str = "day", config: dict | None = None):
    """Ingest OHLCV data as Parquet and track it with DVC."""
    if config is None:
        config = load_config("data")

    if freq == "all":
        freqs = config.get("frequencies", ["day"])
    else:
        freqs = [freq]

    ensure_dvc_repo()

    results: dict[str, list[str]] = {}
    for f in freqs:
        paths = []
        for ticker in config["tickers"]:
            path = fetch_and_store(
                ticker,
                config["start_date"],
                config["end_date"],
                f,
            )
            paths.append(str(path))
        results[f] = paths
    remove_checkpoints()
    return results

@flow(persist_result=True, result_storage=CHECKPOINT_STORAGE)
def train(config: dict | None = None):
    """Example training flow using optuna settings."""
    if config is None:
        config = load_config("optuna")
    print(f"Training models with search spaces: {list(config.keys())}")
    remove_checkpoints()

@flow(persist_result=True, result_storage=CHECKPOINT_STORAGE)
def backtest():
    """Dummy backtesting flow"""
    print("Running backtests...")
    remove_checkpoints()


@flow(persist_result=True, result_storage=CHECKPOINT_STORAGE)
def run_all(freq: str = "daily", do_cleanup: bool = False):
    """Run ingest, train and backtest flows sequentially, then optionally cleanup"""
    data_cfg = load_config("data")
    optuna_cfg = load_config("optuna")
    ingest(freq, config=data_cfg)
    ensure_disk_space()
    train(config=optuna_cfg)
    ensure_disk_space()
    backtest()
    if do_cleanup:
        cleanup_flow()
    remove_checkpoints()


@flow(persist_result=True, result_storage=CHECKPOINT_STORAGE)
def feature_build(freq: str = "day", exogenous: dict[str, str] | None = None):
    """Build engineered features from Parquet price data."""
    if exogenous is None:
        exogenous = {}

    if freq == "all":
        freqs = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
    else:
        freqs = [freq]

    ensure_dvc_repo()

    results: dict[str, list[str]] = {}
    for f in freqs:
        paths = []
        for src in (DATA_DIR / f).glob("*.parquet"):
            exo_path = exogenous.get(src.stem)
            exo = Path(exo_path) if exo_path else None
            dest = build_features(src, exogenous=exo)
            paths.append(str(dest))
        results[f] = paths
    remove_checkpoints()
    return results


