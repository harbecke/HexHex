"""Temperature schedules for move sampling.

A schedule is a callable mapping a 0-based move index to a temperature `T`:
- `T = 0`     → greedy argmax over logits
- `T = inf`   → uniform random over board cells (logits ignored)
- `T > 0`     → sample from softmax(logits / T)

`random_first_moves` lets any schedule open with `k` uniform-random moves
before the underlying schedule kicks in (move `k` is the schedule's move 0).
"""

import math
from typing import Iterable, Tuple

import torch
from torch.distributions.categorical import Categorical


class TemperatureSchedule:
    def __init__(self, random_first_moves: int = 0):
        self.random_first_moves = int(random_first_moves)

    def __call__(self, move_idx: int) -> float:
        if move_idx < self.random_first_moves:
            return math.inf
        return self._post_opening(move_idx - self.random_first_moves)

    def _post_opening(self, k: int) -> float:
        raise NotImplementedError


class Fixed(TemperatureSchedule):
    def __init__(self, value: float, random_first_moves: int = 0):
        super().__init__(random_first_moves)
        self.value = float(value)

    def _post_opening(self, k):
        return self.value


class ExpDecay(TemperatureSchedule):
    def __init__(self, base: float, decay: float, random_first_moves: int = 0):
        super().__init__(random_first_moves)
        self.base = float(base)
        self.decay = float(decay)

    def _post_opening(self, k):
        return self.base * self.decay ** k


class Step(TemperatureSchedule):
    """Piecewise-constant schedule. `steps` is a list of (move_threshold, T)
    pairs; at move `k` the temperature is the `T` of the largest threshold
    `<= k`. AlphaGo uses two phases, e.g. [(0, 1.0), (30, 0.0)]."""

    def __init__(self, steps: Iterable[Tuple[int, float]], random_first_moves: int = 0):
        super().__init__(random_first_moves)
        self.steps = sorted(((int(t), float(v)) for t, v in steps), key=lambda s: s[0])
        if not self.steps:
            raise ValueError("step schedule needs at least one (threshold, temperature) pair")

    def _post_opening(self, k):
        t = self.steps[0][1]
        for threshold, temp in self.steps:
            if k >= threshold:
                t = temp
            else:
                break
        return t


def from_config(cfg) -> TemperatureSchedule:
    """Build a schedule from an OmegaConf dict.

    Required: `type` ∈ {fixed, exp_decay, step}.
    Optional: `random_first_moves` (default 0).
    Type-specific:
      fixed     — value
      exp_decay — base, decay
      step      — steps (list of [threshold, temperature] pairs)
    """
    rfm = int(cfg.get('random_first_moves', 0))
    schedule_type = cfg.type
    if schedule_type == 'fixed':
        return Fixed(value=cfg.value, random_first_moves=rfm)
    if schedule_type == 'exp_decay':
        return ExpDecay(base=cfg.base, decay=cfg.decay, random_first_moves=rfm)
    if schedule_type == 'step':
        return Step(steps=cfg.steps, random_first_moves=rfm)
    raise ValueError(f"unknown temperature schedule type: {schedule_type!r}")


def select_moves(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if math.isinf(temperature):
        n = logits.shape[1]
        return torch.randint(0, n, (logits.shape[0],), device=logits.device)
    if temperature < 1e-10:
        return logits.argmax(1)
    return Categorical(logits=logits / temperature).sample()
