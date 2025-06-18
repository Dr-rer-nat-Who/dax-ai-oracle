from pathlib import Path
import subprocess
import sys

import pandas as pd
import yaml
import yfinance as yf
from prefect import flow, task
from prefect.filesystems import LocalFileSystem
from prefect.runtime.flow_run import FlowRunContext


sys.path.append(str(Path(__file__).resolve().parents[1]))
from features.pipelines import compute_features
from .cleanup import _resample_5min, cleanup_flow

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


def _maybe_call(task_func, *args, **kwargs):
    """Call a Prefect task inside a flow or fall back to the raw function."""
    if FlowRunContext.get():
        return task_func(*args, **kwargs)
    if hasattr(task_func, "fn"):
        return task_func.fn(*args, **kwargs)
    return task_func(*args, **kwargs)


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

    dest_dir = DATA_DIR / "raw" / freq
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{ticker}.parquet"

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    if freq == "minute" and path.exists():
        existing = pd.read_parquet(path)
        if not existing.empty:
            start_ts = existing.index.max() + pd.Timedelta(minutes=1)
    else:
        existing = pd.DataFrame()

    if freq == "minute" and (end_ts - start_ts) > pd.Timedelta(days=30):
        chunks: list[pd.DataFrame] = []
        s = start_ts
        while s < end_ts:
            e = min(s + pd.Timedelta(days=30), end_ts)
            chunk = yf.download(
                ticker,
                start=s.isoformat(),
                end=e.isoformat(),
                interval=interval,
                progress=False,
            )
            chunks.append(chunk)
            s = e
        new = pd.concat(chunks) if chunks else pd.DataFrame()
        df = pd.concat([existing, new])
    else:
        new = yf.download(
            ticker,
            start=start_ts.isoformat(),
            end=end_ts.isoformat(),
            interval=interval,
            progress=False,
        )
        df = pd.concat([existing, new]) if not existing.empty else new

    if freq == "minute":
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
        older = df[df.index < cutoff]
        recent = df[df.index >= cutoff]
        if not older.empty:
            older = _resample_5min(older)
        df = pd.concat([older, recent]).sort_index()

    df = df[~df.index.duplicated(keep="last")]
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

    _maybe_call(ensure_dvc_repo)

    results: dict[str, list[str]] = {}
    for f in freqs:
        range_cfg = config.get("date_ranges", {}).get(f, {})
        start = range_cfg.get("start_date", config.get("start_date"))
        end = range_cfg.get("end_date", config.get("end_date"))

        paths = []
        for ticker in config["tickers"]:
            path = _maybe_call(
                fetch_and_store,
                ticker,
                start,
                end,
                f,
            )
            paths.append(str(path))
        results[f] = paths
    _maybe_call(remove_checkpoints)
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
    _maybe_call(ensure_disk_space)
    train(config=optuna_cfg)
    _maybe_call(ensure_disk_space)
    backtest()
    if do_cleanup:
        cleanup_flow()
    _maybe_call(remove_checkpoints)


@flow(persist_result=True, result_storage=CHECKPOINT_STORAGE)
def feature_build(freq: str = "day", exogenous: dict[str, str] | None = None):
    """Build engineered features from Parquet price data."""
    if exogenous is None:
        exogenous = {}

    if freq == "all":
        freqs = [d.name for d in (DATA_DIR / "raw").iterdir() if d.is_dir()]
    else:
        freqs = [freq]

    _maybe_call(ensure_dvc_repo)

    results: dict[str, list[str]] = {}
    for f in freqs:
        paths = []
        for src in (DATA_DIR / "raw" / f).glob("*.parquet"):
            exo_path = exogenous.get(src.stem)
            exo = Path(exo_path) if exo_path else None
            dest = _maybe_call(build_features, src, exogenous=exo)
            paths.append(str(dest))
        results[f] = paths
    _maybe_call(remove_checkpoints)
    return results


