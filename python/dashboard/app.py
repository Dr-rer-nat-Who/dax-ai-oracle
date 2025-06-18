from __future__ import annotations

import pickle
import time
from pathlib import Path

import pandas as pd
import streamlit as st

try:  # optional, can be missing in test environment
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency
    yf = None

try:  # optional dependency for equity curves
    import vectorbt as vbt
except Exception:  # pragma: no cover - optional dependency
    vbt = None

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "python" / "models"
BEST_DIR = ROOT / "mlruns" / "best"
REFRESH_SEC = 60


def auto_refresh(interval: int = REFRESH_SEC) -> None:
    """Rerun the script every ``interval`` seconds."""
    if "_last_refresh" not in st.session_state:
        st.session_state["_last_refresh"] = time.time()
    if time.time() - st.session_state["_last_refresh"] > interval:
        st.session_state["_last_refresh"] = time.time()
        st.experimental_rerun()


def show_live() -> None:
    """Display real-time ticker data and signals."""
    auto_refresh()
    st.header("Live View")
    if yf is not None:
        try:
            data = yf.download("^GDAXI", period="1d", interval="1m")
            st.line_chart(data["Close"])
        except Exception as exc:  # pragma: no cover - network issues
            st.warning(f"Could not download data: {exc}")
    else:  # pragma: no cover - optional dependency
        st.info("yfinance not available")

    signals_path = MODELS_DIR / "signals.pkl"
    if signals_path.exists():
        signals = pd.read_pickle(signals_path)
        st.subheader("Current Signals")
        st.dataframe(signals.tail())
    else:
        st.info("No signal file found")


def show_leaderboard() -> None:
    """Display leaderboard from best MLflow runs."""
    st.header("Leaderboard")
    if not BEST_DIR.exists():
        st.info("No best runs directory found")
        return

    dfs = []
    for csv_path in BEST_DIR.rglob("metrics.csv"):
        df = pd.read_csv(csv_path)
        df["model"] = csv_path.parent.name
        dfs.append(df)
    if not dfs:
        st.info("No metrics available")
        return
    leaderboard = pd.concat(dfs)
    if "final_score" in leaderboard.columns:
        leaderboard = leaderboard.sort_values("final_score", ascending=False)
    st.dataframe(leaderboard)


def show_equity() -> None:
    """Plot vectorbt equity curves."""
    st.header("Equity Curves")
    if vbt is None:  # pragma: no cover - optional dependency
        st.info("vectorbt not installed")
        return
    port_files = MODELS_DIR.glob("equity_*.pkl")
    for path in port_files:
        try:
            pf = vbt.Portfolio.load(path)
            fig = pf.total_return().vbt.plot()
            st.plotly_chart(fig)
        except Exception as exc:  # pragma: no cover - corrupted file
            st.warning(f"Failed to load {path.name}: {exc}")


def show_explain() -> None:
    """Show LightGBM importances and TFT attention heatmaps."""
    st.header("Model Explainability")
    imp_path = MODELS_DIR / "lightgbm_importances.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path)
        st.subheader("LightGBM Feature Importance")
        st.bar_chart(imp.set_index("feature")["importance"])
    else:
        st.info("No LightGBM importances found")

    attn_path = MODELS_DIR / "tft_attention.pkl"
    if attn_path.exists():
        attn = pickle.load(open(attn_path, "rb"))
        st.subheader("TFT Attention Heatmap")
        st.image(attn)
    else:
        st.info("No TFT attention heatmap found")


PAGES = {
    "Live": show_live,
    "Leaderboard": show_leaderboard,
    "Equity": show_equity,
    "Explain": show_explain,
}


def main() -> None:
    st.set_page_config(page_title="DAX Dashboard", layout="wide")
    page = st.sidebar.radio("Pages", list(PAGES))
    PAGES[page]()


if __name__ == "__main__":
    main()
