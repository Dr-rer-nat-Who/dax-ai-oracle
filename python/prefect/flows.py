from pathlib import Path
import subprocess

import pandas as pd
import yaml
import yfinance as yf
from prefect import flow, task

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "python" / "data"


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

@flow
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
    return results

@flow
def train(config: dict | None = None):
    """Example training flow using optuna settings."""
    if config is None:
        config = load_config("optuna")
    print(f"Training models with search spaces: {list(config.keys())}")

@flow
def backtest():
    """Dummy backtesting flow"""
    print("Running backtests...")

@flow
def cleanup(config: dict | None = None):
    """Example cleanup flow that uses cleanup settings."""
    if config is None:
        config = load_config("cleanup")
    print(
        f"Cleaning up artifacts older than {config['retention_days']} days (min freq: {config['min_freq']})"
    )

@flow
def run_all(freq: str = "daily", do_cleanup: bool = False):
    """Run ingest, train and backtest flows sequentially, then optionally cleanup"""
    data_cfg = load_config("data")
    optuna_cfg = load_config("optuna")
    cleanup_cfg = load_config("cleanup")

    ingest(freq, config=data_cfg)
    train(config=optuna_cfg)
    backtest()
    if do_cleanup:
        cleanup(config=cleanup_cfg)
