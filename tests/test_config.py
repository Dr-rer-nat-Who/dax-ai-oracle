from daxai.prefect import flows


def test_load_configs():
    cfg = flows.load_config("data")
    assert "tickers" in cfg
    cfg2 = flows.load_config("optuna")
    assert isinstance(cfg2, dict)
    assert "n_linear" in cfg2
