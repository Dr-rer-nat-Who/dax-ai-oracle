"""Label generation utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def label_B1(df: pd.DataFrame) -> pd.Series:
    """Binary up/down label for next step."""
    diff = df["Close"].shift(-1) - df["Close"]
    return (diff > 0).astype(int)


def label_T3(df: pd.DataFrame, thresh: float = 0.002) -> pd.Series:
    """Three-class label with +/- threshold."""
    ret = df["Close"].shift(-1) / df["Close"] - 1
    labels = np.where(ret > thresh, 1, np.where(ret < -thresh, -1, 0))
    return pd.Series(labels, index=df.index)


def label_R(df: pd.DataFrame) -> pd.Series:
    """One-step log return in percent."""
    ret = np.log(df["Close"].shift(-1) / df["Close"]) * 100
    return ret.fillna(0)
