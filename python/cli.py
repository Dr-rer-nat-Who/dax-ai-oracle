import argparse
from prefect import flow

from prefect.flows import run_all, ingest, train, backtest, cleanup


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Command line interface for Prefect flows"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run-all command
    parser_run = subparsers.add_parser(
        "run-all", help="Run ingestion, training and backtesting flows"
    )
    parser_run.add_argument(
        "--freq", default="daily", help="Frequency for ingestion"
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
        "--freq", default="daily", help="Frequency for ingestion"
    )

    # train command
    subparsers.add_parser("train", help="Run training flow")

    # backtest command
    subparsers.add_parser("backtest", help="Run backtesting flow")

    # cleanup command
    subparsers.add_parser("cleanup", help="Run cleanup flow")

    args = parser.parse_args(argv)

    if args.command == "run-all":
        run_all(freq=args.freq, do_cleanup=args.cleanup == "yes")
    elif args.command == "ingest":
        ingest(freq=args.freq)
    elif args.command == "train":
        train()
    elif args.command == "backtest":
        backtest()
    elif args.command == "cleanup":
        cleanup()


if __name__ == "__main__":
    main()
