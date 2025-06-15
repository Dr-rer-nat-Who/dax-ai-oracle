import numpy as np
import pandas as pd

from python.features import pipelines as p


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
        index=pd.date_range("2021-01-01", periods=3, freq="D"),
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
    assert len(feats) == len(df)

