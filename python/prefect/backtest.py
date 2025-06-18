"""Model backtesting utilities using vectorbt and backtrader."""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

try:
    import vectorbt as vbt
except Exception:  # pragma: no cover - optional dependency
    vbt = None

try:
    import backtrader as bt
except Exception:  # pragma: no cover - optional dependency
    bt = None

from prefect import flow, task, get_run_logger

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "python" / "models"
MLRUNS_DIR = ROOT_DIR / "mlruns"
BEST_DIR = MLRUNS_DIR / "best"
DATA_DIR = ROOT_DIR / "python" / "data"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _make_price_data(n: int = 200) -> pd.DataFrame:
    """Create synthetic OHLC price dataframe."""
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(scale=1.0, size=n))
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "Open": close,
        "High": close,
        "Low": close,
        "Close": close,
    }, index=index)
    return df


def _make_features(prices: pd.DataFrame, n_feats: int) -> pd.DataFrame:
    """Generate simple lagged return features."""
    rets = prices["Close"].pct_change().fillna(0)
    feats = [rets.shift(i).fillna(0) for i in range(n_feats)]
    X = pd.concat(feats, axis=1)
    X.columns = [f"lag_{i}" for i in range(n_feats)]
    return X


def _predict(model: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    weights = np.asarray(model.get("weights"))
    return X.iloc[:, : len(weights)].to_numpy().dot(weights)


def _find_model_paths(mlruns_dir: Path) -> List[Path]:
    """Return list of model artifact paths under an MLflow ``mlruns`` directory."""
    if not mlruns_dir.exists():
        return []
    paths: List[Path] = []
    for p in mlruns_dir.rglob("*.pkl"):
        if "best" in p.parts:
            continue
        if p.name == "model.pkl" or p.suffix == ".pkl":
            paths.append(p)
    return paths


def _vectorbt_metrics(prices: pd.DataFrame, preds: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    """Run vectorbt screening and return metrics and trading signals."""
    if vbt is None:  # pragma: no cover - optional dependency
        raise RuntimeError("vectorbt not installed")
    signals = np.sign(preds)
    pf = vbt.Portfolio.from_signals(
        prices["Close"],
        entries=signals > 0,
        exits=signals < 0,
        slippage=0.0001,
        fees=0.0,
    )
    y_true = (prices["Close"].diff().shift(-1) > 0).astype(int)[:-1]
    y_pred = (preds > 0).astype(int)[:-1]
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, bal_acc, f1, signals


if bt is not None:
    class _MLStrategy(bt.Strategy):  # type: ignore[misc]
        params: dict = dict(signals=None)

        def __init__(self):
            self.i = 0

        def next(self):  # pragma: no cover - relies on backtrader
            sig = self.p.signals[self.i]
            if not self.position:
                if sig > 0:
                    self.buy()
                elif sig < 0:
                    self.sell()
            else:
                if sig == 0:
                    self.close()
                elif sig > 0 and self.position.size < 0:
                    self.close(); self.buy()
                elif sig < 0 and self.position.size > 0:
                    self.close(); self.sell()
            self.i += 1
else:  # pragma: no cover - backtrader optional
    class _MLStrategy:
        """Fallback when backtrader is missing."""

        params: dict = dict(signals=None)

        def __init__(self):
            self.i = 0

        def next(self) -> None:
            self.i += 1


def _backtrader_metrics(prices: pd.DataFrame, signals: np.ndarray) -> Tuple[float, float, float, float, pd.Series]:
    """Run a simple backtrader simulation and compute performance stats."""
    if bt is None:  # pragma: no cover - optional dependency
        raise RuntimeError("backtrader not installed")

    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=prices)
    cerebro.adddata(data)
    cerebro.addstrategy(_MLStrategy, signals=signals)
    cerebro.broker.set_slippage_perc(0.0005)
    cerebro.broker.setcommission(commission=0.0004)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="returns")
    res = cerebro.run()[0]
    returns = pd.Series(res.analyzers.returns.get_analysis())
    tot_ret = (1 + returns).prod() - 1
    ann_ret = (1 + tot_ret) ** (252 / len(returns)) - 1
    mean = returns.mean() * 252
    std = returns.std() * np.sqrt(252)
    sharpe = mean / std if std != 0 else 0.0
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = mean / downside if downside != 0 else 0.0
    cum = (1 + returns).cumprod()
    mdd = ((cum.cummax() - cum) / cum.cummax()).max()
    return float(sharpe), float(sortino), float(mdd), float(ann_ret), cum


# ---------------------------------------------------------------------------
# Prefect tasks and flow
# ---------------------------------------------------------------------------

@task(log_prints=True)
def backtest_model(path: Path, prices: pd.DataFrame) -> Dict[str, Any]:
    try:
        logger = get_run_logger()
    except Exception:  # pragma: no cover - outside flow context
        import logging

        logger = logging.getLogger(__name__)
    logger.info("Backtesting %s", path.name)
    with open(path, "rb") as f:
        model = pickle.load(f)
    n_feats = len(model.get("weights", []))
    X = _make_features(prices, n_feats)
    preds = _predict(model, X)
    acc, bal_acc, f1, signals = _vectorbt_metrics(prices, preds)
    sharpe, sortino, mdd, cagr, equity = _backtrader_metrics(prices, signals)
    final_score = 0.4 * bal_acc + 0.6 * min(sharpe / 3, 1)
    metrics = dict(
        accuracy=acc,
        balanced_accuracy=bal_acc,
        f1=f1,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=mdd,
        cagr=cagr,
        final_score=final_score,
    )
    logger.info("Metrics for %s: %s", path.name, metrics)
    return metrics, equity


@flow
def backtest(
    mlruns_dir: Path | None = None,
    best_dir: Path | None = None,
    data_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run backtests for model artifacts stored in ``mlruns``."""
    if mlruns_dir is None:
        mlruns_dir = MLRUNS_DIR
    if best_dir is None:
        best_dir = BEST_DIR
    if data_dir is None:
        data_dir = DATA_DIR
    best_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    model_paths = _find_model_paths(mlruns_dir)
    for freq_dir in [d for d in data_dir.iterdir() if d.is_dir()]:
        for price_file in sorted(freq_dir.glob("*.parquet")):
            prices = pd.read_parquet(price_file)
            for model_path in model_paths:
                metrics, equity = backtest_model.fn(model_path, prices)
                metrics["model"] = model_path.stem
                metrics["data"] = price_file.stem
                metrics["frequency"] = freq_dir.name
                results.append(metrics)
                if metrics["final_score"] >= 0.6:
                    dest = best_dir / model_path.stem
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(model_path, dest / model_path.name)
                    pd.DataFrame([metrics]).to_csv(
                        dest / f"{price_file.stem}_metrics.csv", index=False
                    )
                    equity.to_csv(dest / f"{price_file.stem}_equity.csv", header=False)
    if results:
        pd.DataFrame(results).to_csv(best_dir / "metrics.csv", index=False)
    return results


__all__ = ["backtest", "backtest_model"]
