from pathlib import Path

import yaml
from prefect import flow

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(name: str) -> dict:
    """Load a YAML config by name from the configs directory."""
    with open(CONFIG_DIR / f"{name}.yaml", "r") as f:
        return yaml.safe_load(f)

@flow
def ingest(freq: str = "daily", config: dict | None = None):
    """Example ingestion flow that uses settings from data.yaml."""
    if config is None:
        config = load_config("data")
    print(
        f"Ingesting {config['tickers']} from {config['start_date']} to {config['end_date']} with frequency: {freq}"
    )

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
