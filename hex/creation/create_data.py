#!/usr/bin/env python
import numpy as np
import torch

from hex.logic.hexboard import Board
from hex.logic.hexgame import MultiHexGame
from hex.utils import utils
from hex.utils.logger import logger


class SelfPlayGenerator:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.board_size = model.board_size

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
            device=utils.device,
            noise=self.args.get('noise'),
            noise_parameters=[float(parameter) for parameter in self.args.get('noise_parameters').split(",")],
            temperature=self.args.getfloat('temperature'),
            temperature_decay=self.args.getfloat('temperature_decay')
        )
        board_states, moves, targets = multihexgame.play_moves()

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
    for file_idx in range(args.getint('data_range_min'), args.getint('data_range_max')):
        all_boards_tensor = torch.Tensor()
        all_moves = torch.LongTensor()
        all_results = torch.Tensor()
        for _ in range(args.getint('samples_per_file')):
            board_tensor, move, result = next(position_generator)
            all_boards_tensor = torch.cat((all_boards_tensor, board_tensor.unsqueeze(0)))
            all_moves = torch.cat((all_moves, move.unsqueeze(0)))
            all_results = torch.cat((all_results, result.unsqueeze(0)))

        def k_th_move_idx(k):
            return [idx for idx in range(all_boards_tensor.shape[0]) if torch.sum(all_boards_tensor[idx, :2]) == k]

        first_move_indices = k_th_move_idx(0)
        first_move_frequency = torch.zeros([board_size ** 2], dtype=torch.int32)
        for x in first_move_indices:
            first_move_frequency[all_moves[x].item()] += 1
        logger.info("First move frequency:\n" + str(first_move_frequency.view(board_size, board_size).numpy()) + '\n')

        with torch.no_grad():
            board = Board(model.board_size)
            ratings = model(board.board_tensor.unsqueeze(0)).view(board_size, board_size)
            with np.printoptions(precision=1, suppress=True):
                logger.info("First move ratings\n" + str(ratings.numpy()))

        file_name = f'data/{args.get("run_name")}_{file_idx}.pt'
        torch.save((all_boards_tensor, all_moves, all_results), file_name)
        logger.info(f'self-play data generation wrote {file_name}')

