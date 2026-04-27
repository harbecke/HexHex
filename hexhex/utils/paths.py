"""Path resolution for experiment runs and reference models.

Each training run owns one directory `runs/<exp_id>/` containing everything:
    runs/<exp_id>/
        .hydra/                         resolved config, overrides
        repeated_self_training.log      Hydra-managed log file
        ratings.txt                     final ELO ranking
        models/<auto_model_name>_NNNN.pt  per-iteration checkpoints
        data.pt                         optional saved replay buffer
        events.out.tfevents.*           TensorBoard events at run root

`exp_id` has the form `<date>_<time>_<preset>_<auto_model_name>[_<exp_name>]`
and is constructed by Hydra's `hydra.run.dir` interpolation in conf/config.yaml,
so all artifacts share one timestamp.

Reference models live in `reference_models/` and are loaded by name. There is no
fallback — a missing reference model raises FileNotFoundError.
"""
import os
from pathlib import Path

from omegaconf import OmegaConf

RUNS_DIR = "runs"
REFERENCE_MODELS_DIR = "reference_models"

_run_dir: str | None = None


def auto_model_name(model_cfg) -> str:
    return f"b{model_cfg.board_size}_l{model_cfg.layers}_c{model_cfg.intermediate_channels}"


def _suffix(value) -> str:
    return f"_{value}" if value else ""


OmegaConf.register_new_resolver("suffix", _suffix, replace=True)


def set_run_dir(run_dir: str) -> None:
    global _run_dir
    _run_dir = run_dir
    Path(run_models_dir()).mkdir(parents=True, exist_ok=True)


def get_run_dir() -> str:
    if _run_dir is None:
        raise RuntimeError("run dir not set; call set_run_dir() first")
    return _run_dir


def run_models_dir() -> str:
    return os.path.join(get_run_dir(), "models")


def run_model_path(name: str) -> str:
    return os.path.join(run_models_dir(), f"{name}.pt")


def run_data_path() -> str:
    return os.path.join(get_run_dir(), "data.pt")


def saved_data_path(exp_id: str) -> str:
    return os.path.join(RUNS_DIR, exp_id, "data.pt")


def reference_model_path(name: str) -> str:
    candidate = os.path.join(REFERENCE_MODELS_DIR, f"{name}.pt")
    if not os.path.exists(candidate):
        raise FileNotFoundError(
            f"reference model '{name}' not found at {candidate}"
        )
    return candidate
