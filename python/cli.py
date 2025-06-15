import argparse
from pathlib import Path

import yaml
from prefect.flows import run_all, ingest, train, backtest, cleanup

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
        "run-all", help="Run ingestion, training and backtesting flows"
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

    # train command
    subparsers.add_parser("train", help="Run training flow")

    # backtest command
    subparsers.add_parser("backtest", help="Run backtesting flow")

    # cleanup command
    subparsers.add_parser("cleanup", help="Run cleanup flow")

    args = parser.parse_args(argv)

    if args.command == "run-all":
        run_all(
            freq=args.freq,
            do_cleanup=args.cleanup == "yes",
        )
    elif args.command == "ingest":
        ingest(freq=args.freq, config=data_cfg)
    elif args.command == "train":
        train(config=load_config("optuna"))
    elif args.command == "backtest":
        backtest()
    elif args.command == "cleanup":
        cleanup(config=load_config("cleanup"))


if __name__ == "__main__":
    main()
