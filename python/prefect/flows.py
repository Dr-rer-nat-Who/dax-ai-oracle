from prefect import flow

@flow
def ingest(freq: str = "daily"):
    """Dummy ingestion flow"""
    print(f"Ingesting data with frequency: {freq}")

@flow
def train():
    """Dummy training flow"""
    print("Training model...")

@flow
def backtest():
    """Dummy backtesting flow"""
    print("Running backtests...")

@flow
def cleanup():
    """Dummy cleanup flow"""
    print("Cleaning up artifacts...")

@flow
def run_all(freq: str = "daily", do_cleanup: bool = False):
    """Run ingest, train and backtest flows sequentially, then optionally cleanup"""
    ingest(freq)
    train()
    backtest()
    if do_cleanup:
        cleanup()
