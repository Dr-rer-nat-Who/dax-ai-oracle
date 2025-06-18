"""Collection of simple model implementations used in Optuna studies.

<<<<<<< codex/update-run_study-to-use-medianpruner
The individual model modules are imported lazily to avoid optional
dependencies at package import time. ``importlib`` is used elsewhere to
load the required module when a specific model is requested.
"""

=======
>>>>>>> main
__all__ = [
    "lightgbm",
    "catboost",
    "tabnet",
    "prophet",
    "n_linear",
    "lstm",
    "tft",
    "autoformer",
    "informer",
    "patchtst",
    "timesnet",
    "finrl_ppo",
]


def __getattr__(name):
    if name in __all__:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__} has no attribute {name}")
