from daxai.prefect import flows, cleanup


def test_flows_have_checkpoints():
    assert flows.ingest.persist_result is True
    assert flows.ingest.result_storage == flows.CHECKPOINT_STORAGE
    assert flows.train.persist_result is True
    assert flows.backtest.persist_result is True
    assert flows.run_all.persist_result is True
    assert flows.feature_build.persist_result is True


def test_remove_checkpoints(tmp_path):
    base = cleanup.CHECKPOINT_BASE
    if base.exists():
        for p in base.iterdir():
            if p.is_dir():
                for sub in p.iterdir():
                    if sub.is_file():
                        sub.unlink()
                p.rmdir()
    base.mkdir(exist_ok=True)
    (base / "manual").mkdir()
    assert any(base.iterdir())
    cleanup.remove_checkpoints.fn()
    assert not any(base.iterdir())
