import itertools
from collections import defaultdict

import torch

from hexhex.evaluation import evaluate_two_models
from hexhex.logic.hexboard import Board
from hexhex.logic.hexgame import MultiHexGame
from hexhex.utils.logger import logger
from hexhex.utils.summary import writer
from hexhex.utils.utils import load_model


class TestModel:
    def __init__(self, move_ids):
        self.move_ids = move_ids

    def __call__(self, x):
        y = x[0]
        possible_moves = (1 - y[0] - y[1]).view(-1)
        num_moves = int(sum((y[0] + y[1]).view(-1)))
        move = self.move_ids[num_moves//2]
        out = torch.zeros_like(y[0].view(-1))
        move_idx = possible_moves.nonzero()[move]
        out[move_idx] = 1000
        return out.unsqueeze(0)


def win_count_3(model_name):
    model = load_model(f'models/{model_name}.pt')
    board_size = model.board_size

    lose_count = 0
    game_count = 0

    for test_model_starts in [True, False]:
        if test_model_starts:
            move_generator = [[0, 1, 7, 8]] + [range(x) for x in [7, 5, 3, 1]]
        else:
            move_generator = [range(x) for x in [8, 6, 4, 2]]
        for xs in itertools.product(*move_generator):
            board = Board(board_size)
            counter_model = TestModel(xs)
            models = (counter_model, model) if test_model_starts else (model, counter_model)
            game = MultiHexGame((board,), models, noise=None, noise_parameters=None,
                temperature=0, temperature_decay=0)
            game.play_moves()
            if board.winner == [1 - int(test_model_starts)]:
                lose_count += 1
            game_count += 1
    logger.info(f"Lost {lose_count} / {game_count} games")


def win_count(model_name, reference_models, cfg, verbose):
    if verbose:
        opponents = ', '.join(reference_models.keys())
        logger.info(f"Evaluating {model_name} vs {opponents} ({cfg.num_opened_moves} opened moves)")

    model = load_model(f'models/{model_name}.pt')
    board_size = model.board_size
    results = defaultdict(lambda: defaultdict(int))

    for opponent_name, opponent_model in reference_models.items():
        result, _ = evaluate_two_models.play_games(
            models=(model, opponent_model),
            num_opened_moves=cfg.num_opened_moves,
            number_of_games=cfg.num_games // 2,
            batch_size=cfg.batch_size,
            temperature=cfg.temperature,
            temperature_decay=cfg.temperature_decay,
            plot_board=cfg.plot_board
        )

        results[model_name][opponent_name] = result[0][0] + result[1][0]
        results[opponent_name][model_name] = result[0][1] + result[1][1]

        lose_count = results[opponent_name][model_name]
        game_count = results[model_name][opponent_name] + results[opponent_name][model_name]

        if verbose:
            win_count_ = game_count - lose_count
            win_rate = win_count_ / game_count
            logger.info(f"Won {win_count_:4} / {game_count:4} ({round(win_rate * 100):3}%) games against {opponent_name}")
            writer.add_scalar(f'lose_rate/{opponent_name}', 1 - win_rate)

    return results
