import argparse
from pathlib import Path
import subprocess

import yaml
from python.prefect.flows import ingest, feature_build, backtest
from python.prefect.train_and_evaluate import train_all
from python.prefect.cleanup import cleanup

ROOT_DIR = Path(__file__).resolve().parents[1]

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml", "r") as f:
        return yaml.safe_load(f)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Command line interface for Prefect flows"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run-all command
    data_cfg = load_config("data")
    default_freq = data_cfg.get("frequencies", ["daily"])[-1]

    parser_run = subparsers.add_parser(
        "run-all",
        help="Run the full pipeline and launch the dashboard",
    )
    parser_run.add_argument(
        "--freq", default=default_freq, help="Frequency for ingestion"
    )
    parser_run.add_argument(
        "--cleanup",
        choices=["yes", "no"],
        default="no",
        help="Whether to clean up artifacts after running",
    )

    # ingest command
    parser_ingest = subparsers.add_parser("ingest", help="Run ingestion flow")
    parser_ingest.add_argument(
        "--freq", default=default_freq, help="Frequency for ingestion"
    )

    # feature-build command
    parser_features = subparsers.add_parser(
        "feature-build", help="Generate engineered features"
    )
    parser_features.add_argument(
        "--freq", default=default_freq, help="Frequency for feature generation"
    )

    # train-and-evaluate command
    subparsers.add_parser(
        "train-and-evaluate", help="Run Optuna studies and log to MLflow"
    )

    # backtest command
    subparsers.add_parser("backtest", help="Run backtesting flow")

    # cleanup command
    subparsers.add_parser("cleanup", help="Run cleanup flow")

    args = parser.parse_args(argv)

    if args.command == "run-all":
        ingest(freq=args.freq, config=data_cfg)
        feature_build(freq=args.freq)
        train_all()
        backtest()
        if args.cleanup == "yes":
            cleanup()
        subprocess.Popen(
            ["streamlit", "run", str(ROOT_DIR / "python" / "dashboard" / "app.py")]
        )
    elif args.command == "ingest":
        ingest(freq=args.freq, config=data_cfg)
    elif args.command == "feature-build":
        feature_build(freq=args.freq)
    elif args.command == "train-and-evaluate":
        train_all()
    elif args.command == "backtest":
        backtest()
    elif args.command == "cleanup":
        cleanup()


if __name__ == "__main__":
    main()
