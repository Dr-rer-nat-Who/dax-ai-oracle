import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from python.features import pipelines as p
from python.features import labels as lbl


def test_window_embeddings_basic():
    returns = pd.Series([0.0, 0.1, -0.1, 0.2])
    emb = p._window_embeddings(returns, 3)
    assert len(emb) == 4
    assert np.isnan(emb.iloc[0]).all()
    assert np.isnan(emb.iloc[1]).all()
    assert len(emb.iloc[2]) == 3
    assert np.isclose(emb.iloc[3].mean(), 0.0, atol=1e-8)


def test_compute_features_with_exogenous(monkeypatch):
    df = pd.DataFrame(
        {
            "Open": [1, 2, 3],
            "High": [1, 2, 3],
            "Low": [1, 2, 3],
            "Close": [1, 2, 3],
        },
        index=pd.date_range("2021-01-01", periods=3, freq="D", tz="UTC"),
    )
    exo = df * 2

    class FakeTalib:
        def ATR(self, high, low, close):
            return pd.Series([0.1] * len(high), index=high.index)

    monkeypatch.setattr(p, "talib", FakeTalib(), raising=False)
    monkeypatch.setattr(p, "_talib_features", lambda df: pd.DataFrame(index=df.index))

    feats = p.compute_features(df, exogenous=exo)
    assert "wick_upper" in feats.columns
    assert "dax_future_spread" in feats.columns
    assert "holiday_flag" in feats.columns
    assert len(feats) == len(df)
    assert feats.index.tz is not None


def test_compute_features_without_talib(monkeypatch):
    df = pd.DataFrame(
        {
            "Open": [1, 2],
            "High": [1, 2],
            "Low": [0, 1],
            "Close": [1, 2],
        },
        index=pd.date_range("2021-01-01", periods=2, freq="D", tz="UTC"),
    )

    monkeypatch.setattr(p, "talib", None, raising=False)
    monkeypatch.setattr(p, "_talib_features", lambda df: pd.DataFrame(index=df.index))

    feats = p.compute_features(df)
    assert list(feats["ATR"]) == [0, 0]
    assert feats.index.tz is not None



def test_label_generation_functions():
    df = pd.DataFrame({"Close": [1.0, 1.02, 0.99, 1.01]})
    b1 = lbl.label_B1(df)
    t3 = lbl.label_T3(df, thresh=0.01)
    r = lbl.label_R(df)
    assert len(b1) == len(df)
    assert set(t3.unique()) <= {1, 0, -1}
    assert np.isfinite(r).all()


def test_eurex_holiday_flag():
    idx = pd.to_datetime(["2021-12-27", "2021-12-29"])
    feats = p._datetime_features(idx)
    assert list(feats["holiday_flag"]) == [1, 0]


def test_compute_features_localizes_naive_index(monkeypatch):
    df = pd.DataFrame(
        {
            "Open": [1, 2],
            "High": [1, 2],
            "Low": [0, 1],
            "Close": [1, 2],
        },
        index=pd.date_range("2022-01-01", periods=2, freq="D"),
    )

    monkeypatch.setattr(p, "talib", None, raising=False)
    monkeypatch.setattr(p, "_talib_features", lambda df: pd.DataFrame(index=df.index))

    feats = p.compute_features(df)
    assert feats.index.tz is not None
    assert str(feats.index.tz) == "UTC"

