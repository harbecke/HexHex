from hexhex.evaluation import evaluate_two_models
from hexhex.logic import temperature
from hexhex.utils.logger import logger
from hexhex.utils.paths import run_model_path
from hexhex.utils.summary import writer
from hexhex.utils.utils import load_model


def win_count(model_name, reference_models, cfg, verbose, step=None):
    """Play `model_name` against each reference model and log win rate per reference.

    Returns a dict mapping reference name -> win rate of `model_name` (0..1).
    """
    if verbose:
        opponents = ', '.join(reference_models.keys())
        logger.info(f"Evaluating {model_name} vs {opponents} ({cfg.num_opened_moves} opened moves)")

    model = load_model(run_model_path(model_name))
    win_rates = {}

    for opponent_name, opponent_model in reference_models.items():
        result, _ = evaluate_two_models.play_games(
            models=(model, opponent_model),
            num_opened_moves=cfg.num_opened_moves,
            number_of_games=cfg.num_games // 2,
            batch_size=cfg.batch_size,
            temperature_schedule=temperature.from_config(cfg.temperature),
            plot_board=cfg.plot_board
        )

        wins = result[0][0] + result[1][0]
        losses = result[0][1] + result[1][1]
        total = wins + losses
        win_rate = wins / total if total else 0.0
        win_rates[opponent_name] = win_rate

        if verbose:
            logger.info(f"Won {wins:4} / {total:4} ({round(win_rate * 100):3}%) games against {opponent_name}")
            writer.add_scalar(f'win_rate/{opponent_name}', win_rate, step)

    return win_rates
