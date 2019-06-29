#!/usr/bin/env python

import argparse
from configparser import ConfigParser

import torch

from hex.logic.hexboard import Board
from hex.logic.hexgame import MultiHexGame
from hex.utils import utils
from hex.utils.logger import logger
from hex.utils.utils import load_model


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
            noise_parameters=self.args.get('noise_parameters'),
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


def generate_data_files(file_counter_start, file_counter_end, samples_per_file, model, device, batch_size, run_name,
                        noise, noise_parameters, temperature, temperature_decay, board_size):
    '''
    generates data files with run_name indexed from file_counter_start to file_counter_end
    samples_per_files number of triples (board, move, target)
    '''
    logger.debug("=== creating data from self play ===")
    all_board_states = torch.Tensor()
    all_moves = torch.LongTensor()
    all_targets = torch.Tensor()

    file_counter = file_counter_start
    while file_counter < file_counter_end:
        while all_board_states.shape[0] < samples_per_file:
            boards = [Board(size=board_size) for _ in range(batch_size)]
            multihexgame = MultiHexGame(boards=boards, models=(model,), device=device, noise=noise,
                noise_parameters=noise_parameters, temperature=temperature, temperature_decay=temperature_decay)
            board_states, moves, targets = multihexgame.play_moves()

            all_board_states = torch.cat((all_board_states,board_states))
            all_moves = torch.cat((all_moves,moves))
            all_targets = torch.cat((all_targets,targets))

        file_name = f'data/{run_name}_{file_counter}.pt'
        torch.save(
                (
                    # clone to avoid large files for large batch sizes
                    # https://stackoverflow.com/questions/46227756/resized-copy-of-pytorch-tensor-dataset
                    all_board_states[:samples_per_file].clone(),
                    all_moves[:samples_per_file].clone(),
                    all_targets[:samples_per_file].clone()
                ),
                file_name)
        logger.info(f'wrote {file_name}')
        file_counter += 1

        all_board_states = all_board_states[samples_per_file:]
        all_moves = all_moves[samples_per_file:]
        all_targets = all_targets[samples_per_file:]
    logger.debug("")


def get_args(config_file):
    config = ConfigParser()
    config.read(config_file)
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_range_min', type=int, default=config.getint('CREATE DATA', 'data_range_min'))
    parser.add_argument('--data_range_max', type=int, default=config.getint('CREATE DATA', 'data_range_max'))
    parser.add_argument('--samples_per_file', type=int, default=config.getint('CREATE DATA', 'samples_per_file'))
    parser.add_argument('--model', type=str, default=config.get('CREATE DATA', 'model'))
    parser.add_argument('--batch_size', type=int, default=config.getint('CREATE DATA', 'batch_size'))
    parser.add_argument('--run_name', type=str, default=config.get('CREATE DATA', 'run_name'))
    parser.add_argument('--noise', type=str, default=config.get('CREATE DATA', 'noise'))
    parser.add_argument('--noise_parameters', type=str, default=config.get('CREATE DATA', 'noise_parameters'))
    parser.add_argument('--temperature', type=float, default=config.getfloat('CREATE DATA', 'temperature'))
    parser.add_argument('--temperature_decay', type=float, default=config.getfloat('CREATE DATA', 'temperature_decay'))
    parser.add_argument('--board_size', type=int, default=config.getint('CREATE DATA', 'board_size'))

    return parser.parse_args()


def main(config_file='config.ini'):
    args = get_args(config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(f'models/{args.model}.pt')

    generate_data_files(args.data_range_min, args.data_range_max, args.samples_per_file, model, device,
                        args.batch_size, args.run_name, args.noise,
                        [float(parameter) for parameter in args.noise_parameters.split(",")], args.temperature,
                        args.temperature_decay, args.board_size)


def create_self_play_data(args, model):
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

        file_name = f'data/{args.get("run_name")}_{file_idx}.pt'
        torch.save((all_boards_tensor, all_moves, all_results), file_name)
        logger.info(f'self-play data generation wrote {file_name}')


if __name__ == '__main__':
    main()
