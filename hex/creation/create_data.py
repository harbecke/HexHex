#!/usr/bin/env python
import numpy as np
import torch

from hex.logic.hexboard import Board
from hex.logic.hexgame import MultiHexGame
from hex.utils import utils
from hex.utils.logger import logger
from hex.utils.summary import writer


class SelfPlayGenerator:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.board_size = model.board_size
        self.best_move_ratings = []

    def self_play_game(self):
        """
        Generates data points from self play.
        yields 3 tensors containing for each move:
        - board
        - move
        - result of game for active player (-1 or 1)
        """
        boards = [Board(size=self.board_size) for _ in range(self.args.getint('batch_size'))]
        multihexgame = MultiHexGame(
            boards=boards,
            models=(self.model,),
            noise=self.args.get('noise'),
            noise_parameters=[float(parameter) for parameter in self.args.get('noise_parameters').split(",")],
            temperature=self.args.getfloat('temperature'),
            temperature_decay=self.args.getfloat('temperature_decay')
        )
        board_states, moves, targets = multihexgame.play_moves()
        self.best_move_ratings += multihexgame.best_move_ratings

        for board_state, move, target in zip(board_states, moves, targets):
            yield board_state, move, target

    def position_generator(self):
        while True:
            for board_tensor, move_tensor, result_tensor in self.self_play_game():
                yield board_tensor, move_tensor, result_tensor
                # TODO implement mirror logic here, for data augmentation
                # mirror_board = torch.flip(board_tensor, dims=(1, 2)).clone()
                # mirror_policy = torch.flip(mcts_policy, dims=(0,)).clone()
                # same_result = result_tensor.clone()
                # yield mirror_board, mirror_policy, same_result


def create_self_play_data(args, model):
    board_size = model.board_size

    logger.info("")
    logger.info("=== creating data from self play ===")
    self_play_generator = SelfPlayGenerator(model, args)
    position_generator = self_play_generator.position_generator()

    samples_per_model = args.getint('samples_per_model')
    all_boards_tensor = torch.zeros((samples_per_model, 2, board_size, board_size), dtype=torch.float)
    all_moves = torch.zeros((samples_per_model, 1), dtype=torch.long)
    all_results = torch.zeros(samples_per_model, dtype=torch.float)

    for sample_idx in range(samples_per_model):
        board_tensor, move, result = next(position_generator)
        all_boards_tensor[sample_idx] = board_tensor
        all_moves[sample_idx] = move
        all_results[sample_idx] = result

    def k_th_move_idx(k):
        return [idx for idx in range(all_boards_tensor.shape[0]) if torch.sum(all_boards_tensor[idx, :2]) == k]

    first_move_indices = k_th_move_idx(0)
    first_move_frequency = torch.zeros([board_size ** 2], dtype=torch.float)
    first_move_win_percentage = torch.zeros([board_size ** 2], dtype=torch.float)

    for x in first_move_indices:
        first_move_frequency[all_moves[x].item()] += 1
        if all_results[x].item() == 1:
            first_move_win_percentage[all_moves[x].item()] += 1
    first_move_win_percentage /= first_move_frequency

    with np.printoptions(precision=2, suppress=True):
        logger.info("First move frequency:\n" + str(first_move_frequency.view(board_size, board_size).numpy()) + '\n')
        logger.info("First move win percentage:\n" + str(first_move_win_percentage.view(board_size, board_size).numpy()) + '\n')

    with torch.no_grad():
        board = Board(model.board_size)
        ratings = model(board.board_tensor.unsqueeze(0).to(utils.device)).view(board_size, board_size)
        with np.printoptions(precision=1, suppress=True):
            logger.info("First move ratings\n" + str(ratings.cpu().numpy()))

    writer.add_histogram('best_move_ratings', torch.cat(self_play_generator.best_move_ratings))

    logger.info(f'created self-play data')
    return all_boards_tensor, all_moves, all_results
