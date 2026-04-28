import math

import pytest
import torch
from omegaconf import OmegaConf

from hexhex.logic.temperature import ExpDecay, Fixed, Step, from_config, select_moves


def test_fixed_returns_constant():
    s = Fixed(value=0.5)
    assert s(0) == 0.5
    assert s(7) == 0.5
    assert s(100) == 0.5


def test_fixed_with_random_first_moves():
    s = Fixed(value=0.5, random_first_moves=2)
    assert s(0) == math.inf
    assert s(1) == math.inf
    assert s(2) == 0.5
    assert s(10) == 0.5


def test_exp_decay_curve():
    s = ExpDecay(base=4.0, decay=0.5)
    assert s(0) == 4.0
    assert s(1) == 2.0
    assert s(2) == 1.0
    assert s(3) == 0.5


def test_exp_decay_with_random_first_moves_resets_index():
    # With random_first_moves=1, the underlying schedule sees k=0 at move_idx=1.
    s = ExpDecay(base=4.0, decay=0.5, random_first_moves=1)
    assert s(0) == math.inf
    assert s(1) == 4.0  # k=0 → base
    assert s(2) == 2.0  # k=1 → base*decay


def test_step_uses_largest_threshold_le_k():
    s = Step(steps=[(0, 1.0), (3, 0.5), (10, 0.0)])
    assert s(0) == 1.0
    assert s(2) == 1.0
    assert s(3) == 0.5
    assert s(9) == 0.5
    assert s(10) == 0.0
    assert s(50) == 0.0


def test_step_unsorted_input_is_sorted():
    s = Step(steps=[(10, 0.0), (0, 1.0), (3, 0.5)])
    assert s(2) == 1.0
    assert s(3) == 0.5
    assert s(10) == 0.0


def test_step_empty_raises():
    with pytest.raises(ValueError):
        Step(steps=[])


def test_step_with_random_first_moves():
    s = Step(steps=[(0, 1.0), (3, 0.0)], random_first_moves=2)
    assert s(0) == math.inf
    assert s(1) == math.inf
    assert s(2) == 1.0  # k=0
    assert s(4) == 1.0  # k=2
    assert s(5) == 0.0  # k=3


def test_from_config_fixed():
    cfg = OmegaConf.create({'type': 'fixed', 'value': 0.7, 'random_first_moves': 1})
    s = from_config(cfg)
    assert isinstance(s, Fixed)
    assert s(0) == math.inf
    assert s(1) == 0.7


def test_from_config_exp_decay():
    cfg = OmegaConf.create({'type': 'exp_decay', 'base': 4.0, 'decay': 0.8})
    s = from_config(cfg)
    assert isinstance(s, ExpDecay)
    assert s(0) == 4.0
    assert s(1) == pytest.approx(3.2)


def test_from_config_step():
    cfg = OmegaConf.create({'type': 'step', 'steps': [[0, 1.0], [5, 0.0]]})
    s = from_config(cfg)
    assert isinstance(s, Step)
    assert s(4) == 1.0
    assert s(5) == 0.0


def test_from_config_random_first_moves_defaults_to_zero():
    cfg = OmegaConf.create({'type': 'fixed', 'value': 0.5})
    s = from_config(cfg)
    assert s.random_first_moves == 0
    assert s(0) == 0.5


def test_from_config_unknown_type_raises():
    cfg = OmegaConf.create({'type': 'mystery', 'value': 1.0})
    with pytest.raises(ValueError, match="unknown temperature schedule type"):
        from_config(cfg)


def test_select_moves_argmax_at_zero_temperature():
    logits = torch.tensor([[0.1, 0.9, 0.5], [3.0, 1.0, 2.0]])
    moves = select_moves(logits, temperature=0.0)
    assert moves.tolist() == [1, 0]


def test_select_moves_uniform_at_inf_temperature_ignores_logits():
    # A degenerate logit vector that would force argmax=2 must NOT bias inf-T sampling.
    logits = torch.full((2000, 5), -1e9)
    logits[:, 2] = 1e9
    torch.manual_seed(0)
    moves = select_moves(logits, temperature=math.inf)
    counts = torch.bincount(moves, minlength=5).tolist()
    # Uniform over 5 cells with 2000 draws → ~400 each. Loose bounds rule out logit-following.
    for c in counts:
        assert 300 < c < 500, f"counts {counts} not uniform"


def test_select_moves_finite_temperature_samples_from_logits():
    # Strong logit on cell 1 → samples should overwhelmingly pick 1 at T=1.
    logits = torch.tensor([[0.0, 100.0, 0.0]])
    logits = logits.expand(1000, 3)
    torch.manual_seed(0)
    moves = select_moves(logits, temperature=1.0)
    assert (moves == 1).all()


def test_select_moves_returns_correct_shape():
    logits = torch.randn(7, 11)
    assert select_moves(logits, temperature=0.0).shape == (7,)
    assert select_moves(logits, temperature=1.0).shape == (7,)
    assert select_moves(logits, temperature=math.inf).shape == (7,)
