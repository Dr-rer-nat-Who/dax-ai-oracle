import pandas as pd

from daxai.prefect import flows


def test_feature_build_creates_feature_files(tmp_path, monkeypatch):
    # Redirect data and feature directories to tmp_path
    flows.DATA_DIR = tmp_path / "data"
    flows.FEATURES_DIR = tmp_path / "features"
    raw_dir = flows.DATA_DIR / "raw" / "day"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # sample input parquet
    df = pd.DataFrame({"Open": [1], "High": [1], "Low": [1], "Close": [1]}, index=pd.date_range("2020-01-01", periods=1, freq="D"))
    input_path = raw_dir / "sample.parquet"
    df.to_parquet(input_path)

    # patch external calls
    monkeypatch.setattr(flows.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(flows, "ensure_dvc_repo", lambda: None)
    monkeypatch.setattr(flows, "remove_checkpoints", lambda: None, raising=False)
    monkeypatch.setattr(flows, "compute_features", lambda data, exogenous=None: data)

    result = flows.feature_build.fn(freq="day")

    out_path = flows.FEATURES_DIR / "day" / "sample.parquet"
    assert out_path.exists(), "Feature file not created"
    assert result == {"day": [str(out_path)]}
