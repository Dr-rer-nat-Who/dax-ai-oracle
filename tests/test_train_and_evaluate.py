from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from python.prefect.train_and_evaluate import run_study, load_config


def test_all_models_train(tmp_path: Path) -> None:
    cfg = load_config("optuna")
    # run a single trial for speed
    for name, space in cfg.items():
        run_study.fn(name, space or {}, n_trials=1)

