"""Run one referee evaluation with the training loop's reference protocol and summarize it.

Only aggregate counts leave this module: no moves, game records or per-game outcome order.
"""
import math

from hexhex.evaluation.evaluate_two_models import play_games
from hexhex.logic import hexboard, temperature
from hexhex.utils.utils import seed_everything


def wilson_interval(wins, games, z=1.96):
    if games == 0:
        return 0.0, 1.0
    p = wins / games
    denom = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denom
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def games_per_evaluation(match_cfg, board_size):
    """Games one evaluation plays; `play_games` caps each color at the number of distinct openings."""
    per_color = match_cfg.num_games // 2
    if match_cfg.num_opened_moves > 0:
        num_openings = sum(1 for _ in hexboard.first_k_moves(board_size, match_cfg.num_opened_moves))
        per_color = min(per_color, num_openings)
    return 2 * per_color


def run_match(candidate, opponent, match_cfg, seed=None):
    """Play `candidate` against `opponent` and return the aggregate result dict.

    `seed=None` samples fresh openings for every evaluation; a fixed seed replays the same openings, so
    repeated evaluations of one model give identical results.
    """
    if seed is not None:
        seed_everything(seed)
    result, _ = play_games(
        models=(candidate, opponent),
        num_opened_moves=match_cfg.num_opened_moves,
        number_of_games=match_cfg.num_games // 2,
        batch_size=match_cfg.batch_size,
        temperature_schedule=temperature.from_config(match_cfg.temperature),
        plot_board=False,
    )
    # result[0]: candidate owned the pre-placed stone; result[1]: candidate made the first decision after
    # the opening (including whether to swap). Within each, [0] counts candidate wins and [1] losses.
    (first_wins, first_losses), (second_wins, second_losses) = result
    wins = first_wins + second_wins
    losses = first_losses + second_losses
    games = wins + losses
    low, high = wilson_interval(wins, games)
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / games, 3) if games else 0.0,
        "ci95": [round(low, 3), round(high, 3)],
        "as_first": {"wins": first_wins, "losses": first_losses},
        "as_second": {"wins": second_wins, "losses": second_losses},
    }
