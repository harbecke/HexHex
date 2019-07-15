import itertools

import torch

from hex.logic.hexboard import Board
from hex.logic.hexgame import MultiHexGame
from hex.utils.logger import logger
from hex.utils.utils import load_model, device


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


def win_count(model_name, config):
    logger.info("Determining win count against test model")

    model = load_model(model_name)
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
            game = MultiHexGame((board,), models, device=device, noise=None,
                                noise_parameters=None, temperature=0, temperature_decay=0)
            game.play_moves()
            if board.winner == [1 - int(test_model_starts)]:
                lose_count += 1
            game_count += 1

    logger.info(f"Lost {lose_count} / {game_count} games")