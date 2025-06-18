"""Cleanup utilities for DAX AI Oracle project."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
from prefect import flow, task, get_run_logger
from prefect.runtime.flow_run import FlowRunContext
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "python" / "data"
CACHE_DIR = Path.home() / ".prefect"
CHECKPOINT_BASE = Path.home() / "checkpoints"
CONFIG_DIR = ROOT_DIR / "python" / "configs"


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml", "r") as f:
        return yaml.safe_load(f)


@task(log_prints=True)
def dvc_gc_workspace() -> None:
    """Run ``dvc gc -w`` to drop unused data objects from the workspace."""
    logger = get_run_logger()
    cmd = ["dvc", "gc", "-w"]
    logger.info("Running %s", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=ROOT_DIR, check=True)
    except Exception as exc:  # pragma: no cover - command may not exist in tests
        logger.error("Failed running dvc gc: %s", exc)


def _resample_5min(df: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    return (
        df.sort_index()
        .resample("5T")
        .agg(agg)
        .dropna(how="any")
    )


@task(log_prints=True)
def drop_unreferenced_parquet() -> None:
    """Delete Parquet files not referenced by Prefect cache and compress old minute data."""
    logger = get_run_logger()

    cfg = load_config("cleanup")
    agg_days = int(cfg.get("aggregate_minute_after_days", 0))
    now = datetime.now()

    cached = set(p.resolve() for p in CACHE_DIR.rglob("*.parquet")) if CACHE_DIR.exists() else set()

    for path in DATA_DIR.rglob("*.parquet"):
        resolved = path.resolve()
        if resolved in cached:
            continue
        if path.parent.name == "minute":
            if agg_days:
                age = now - datetime.fromtimestamp(path.stat().st_mtime)
                if age < timedelta(days=agg_days):
                    continue
            logger.info("Compressing %s to 5-minute bars", path)
            try:
                df = pd.read_parquet(path)
                df_5 = _resample_5min(df)
                df_5.to_parquet(path)
                subprocess.run(["dvc", "add", str(path)], cwd=ROOT_DIR, check=False)
            except Exception as exc:  # pragma: no cover - failures not critical
                logger.error("Failed compressing %s: %s", path, exc)
        else:
            logger.info("Removing unreferenced %s", path)
            try:
                path.unlink(missing_ok=True)
                subprocess.run(["dvc", "remove", str(path)], cwd=ROOT_DIR, check=False)
            except Exception as exc:  # pragma: no cover
                logger.error("Failed removing %s: %s", path, exc)


@task(log_prints=True)
def remove_checkpoints() -> None:
    """Remove checkpoint directory for the current flow run if it exists."""
    ctx = FlowRunContext.get()
    run_id = ctx.flow_run.id if ctx else "manual"
    checkpoint_dir = CHECKPOINT_BASE / str(run_id)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)


def _disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 1024 ** 3


@flow
def cleanup() -> None:
    """Cleanup flow triggered when disk space falls below 5 GB."""
    logger = get_run_logger()
    free_gb = _disk_free_gb(ROOT_DIR)
    logger.info("Free disk space: %.2f GB", free_gb)
    if free_gb >= 5:
        logger.info("Skipping cleanup; sufficient disk space")
        return

    dvc_gc_workspace()
    drop_unreferenced_parquet()
    remove_checkpoints()


@flow
def cleanup_flow() -> None:
    """Wrapper flow calling :func:`cleanup`."""
    cleanup()


__all__ = ["cleanup_flow"]

