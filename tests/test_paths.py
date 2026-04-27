import os
import pytest
from omegaconf import OmegaConf

from hexhex.utils import paths
from hexhex.utils.paths import (
    REFERENCE_MODELS_DIR,
    RUNS_DIR,
    _suffix,
    auto_model_name,
    get_run_dir,
    reference_model_path,
    run_data_path,
    run_model_path,
    run_models_dir,
    saved_data_path,
    set_run_dir,
)


@pytest.fixture(autouse=True)
def reset_run_dir():
    """Each test starts with no run dir set."""
    paths._run_dir = None
    yield
    paths._run_dir = None


def test_auto_model_name_uses_NxN_for_board_size():
    cfg = OmegaConf.create({"board_size": 11, "layers": 18, "intermediate_channels": 64})
    assert auto_model_name(cfg) == "11x11_l18_c64"


def test_auto_model_name_small_board():
    cfg = OmegaConf.create({"board_size": 3, "layers": 2, "intermediate_channels": 5})
    assert auto_model_name(cfg) == "3x3_l2_c5"


def test_suffix_resolver_with_value():
    assert _suffix("lr_3e4") == "_lr_3e4"


def test_suffix_resolver_with_none():
    assert _suffix(None) == ""


def test_suffix_resolver_with_empty_string():
    assert _suffix("") == ""


def test_suffix_resolver_registered_in_omegaconf():
    """The resolver is registered at import time so Hydra's run.dir interpolation works."""
    cfg = OmegaConf.create({"x": "${suffix:foo}", "y": "${suffix:${oc.select:missing,''}}"})
    assert cfg.x == "_foo"
    assert cfg.y == ""


def test_get_run_dir_unset_raises():
    with pytest.raises(RuntimeError, match="run dir not set"):
        get_run_dir()


def test_run_model_path_unset_raises():
    with pytest.raises(RuntimeError, match="run dir not set"):
        run_model_path("foo")


def test_set_run_dir_creates_models_subdir(tmp_path):
    run_dir = tmp_path / "runs" / "exp1"
    set_run_dir(str(run_dir))
    assert (run_dir / "models").is_dir()


def test_set_run_dir_idempotent(tmp_path):
    """Calling twice on an existing dir doesn't fail."""
    run_dir = tmp_path / "runs" / "exp1"
    set_run_dir(str(run_dir))
    set_run_dir(str(run_dir))
    assert (run_dir / "models").is_dir()


def test_run_models_dir_returns_models_subdir(tmp_path):
    run_dir = tmp_path / "runs" / "exp1"
    set_run_dir(str(run_dir))
    assert run_models_dir() == os.path.join(str(run_dir), "models")


def test_run_model_path_joins_name_and_extension(tmp_path):
    run_dir = tmp_path / "runs" / "exp1"
    set_run_dir(str(run_dir))
    assert run_model_path("11x11_l18_c64_0042") == os.path.join(
        str(run_dir), "models", "11x11_l18_c64_0042.pt"
    )


def test_run_data_path(tmp_path):
    run_dir = tmp_path / "runs" / "exp1"
    set_run_dir(str(run_dir))
    assert run_data_path() == os.path.join(str(run_dir), "data.pt")


def test_saved_data_path_relative_to_runs_dir():
    """saved_data_path resolves any exp_id under runs/, independent of the current run."""
    assert saved_data_path("2026-04-27_15-00-00_dev_3x3_l2_c5") == os.path.join(
        RUNS_DIR, "2026-04-27_15-00-00_dev_3x3_l2_c5", "data.pt"
    )


def test_reference_model_path_returns_existing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ref_dir = tmp_path / REFERENCE_MODELS_DIR
    ref_dir.mkdir()
    (ref_dir / "my_model.pt").write_bytes(b"")
    assert reference_model_path("my_model") == os.path.join(REFERENCE_MODELS_DIR, "my_model.pt")


def test_reference_model_path_missing_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / REFERENCE_MODELS_DIR).mkdir()
    with pytest.raises(FileNotFoundError, match="reference model 'nope' not found"):
        reference_model_path("nope")


def test_reference_model_path_no_legacy_fallback(tmp_path, monkeypatch):
    """A model only present under models/ (legacy) must NOT resolve."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / REFERENCE_MODELS_DIR).mkdir()
    legacy_dir = tmp_path / "models"
    legacy_dir.mkdir()
    (legacy_dir / "old_model.pt").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        reference_model_path("old_model")
