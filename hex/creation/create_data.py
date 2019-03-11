#!/usr/bin/env python

import torch

import argparse
from configparser import ConfigParser

from hex.logic.hexboard import Board, to_move
from hex.model.hexconvolution import MCTSModel
from hex.logic.hexgame import MultiHexGame
from hex.utils.logger import logger
from hex.model.mcts import MCTSSearch
from hex.utils.utils import dotdict


class SelfPlayGenerator:
    def __init__(self, model: MCTSModel, mcts_args):
        self.model = model
        self.mcts_args = mcts_args
        self.board_size = model.board_size

    def self_play_mcts_game(self):
        """
        Generates data files from MCTS runs.
        yields 3 tensors containing for each move:
        - board_tensor
        - mcts policy
        - result of game for active player (-1 or 1)
        """
        all_board_tensors = []
        all_mcts_policies = []

        search = MCTSSearch(self.model, self.mcts_args)
        board = Board(size=self.board_size)
        while not board.winner:
            move_counts, Qs = search.simulate(board)
            temperature_freeze = len(board.move_history) >= self.mcts_args.temperature_freeze
            temperature = 0 if temperature_freeze else self.mcts_args.temperature
            mcts_policy = search.move_probabilities(move_counts, temperature)
            move = search.sample_move(mcts_policy)

            all_board_tensors.append(board.board_tensor.clone())
            all_mcts_policies.append(torch.Tensor(mcts_policy).clone())

            board.set_stone(to_move(move, self.board_size))

        result_first_player = 1 if board.winner == [0] else -1
        current_result = result_first_player

        for board_tensor, mcts_policy in zip(all_board_tensors, all_mcts_policies):
            yield board_tensor, mcts_policy, torch.Tensor([current_result])
            current_result = -current_result

    def position_generator(self):
        while True:
            for board_tensor, mcts_policy, result_tensor in self.self_play_mcts_game():
                yield board_tensor, mcts_policy, result_tensor
                mirror_board = torch.flip(board_tensor, dims=(1, 2)).clone()
                mirror_policy = torch.flip(mcts_policy, dims=(0,)).clone()
                same_result = result_tensor.clone()
                yield mirror_board, mirror_policy, same_result



def generate_data_files(file_counter_start, file_counter_end, samples_per_file, model, device, batch_size, run_name,
                        noise, noise_parameters, temperature, temperature_decay, board_size):
    '''
    generates data files with run_name indexed from file_counter_start to file_counter_end
    samples_per_files number of triples (board, move, target)
    '''
    print("=== creating data from self play ===")
    all_board_states = torch.Tensor()
    all_moves = torch.LongTensor()
    all_targets = torch.Tensor()

    file_counter = file_counter_start
    while file_counter < file_counter_end:
        while all_board_states.shape[0] < samples_per_file:
            boards = [Board(size=board_size) for idx in range(batch_size)]
            multihexgame = MultiHexGame(boards=boards, models=(model,), device=device, noise=noise, noise_parameters=noise_parameters, temperature=temperature, temperature_decay=temperature_decay)
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
        print(f'wrote {file_name}')
        file_counter += 1

        all_board_states = all_board_states[samples_per_file:]
        all_moves = all_moves[samples_per_file:]
        all_targets = all_targets[samples_per_file:]
    print("")

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
    parser.add_argument('--c_puct', type=float, default=config.getfloat('CREATE DATA', 'c_puct'))
    parser.add_argument('--num_mcts_simulations', type=int, default=config.getint('CREATE DATA', 'num_mcts_simulations'))
    parser.add_argument('--mcts_batch_size', type=int, default=config.getint('CREATE DATA', 'mcts_batch_size'))
    parser.add_argument('--n_virtual_loss', type=int, default=config.getint('CREATE DATA', 'n_virtual_loss'))

    return parser.parse_args()

def main(config_file='config.ini'):
    args = get_args(config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('models/{}.pt'.format(args.model), map_location=device)

    if model.__class__ == MCTSModel:
        create_mcts_self_play_data(args, model)

    else:
        generate_data_files(args.data_range_min, args.data_range_max, args.samples_per_file, model, device,
                            args.batch_size,
                            args.run_name, args.noise,
                            [float(parameter) for parameter in args.noise_parameters.split(",")], args.temperature,
                            args.temperature_decay, args.board_size)


def create_mcts_self_play_data(args, model):
    logger.info("=== creating data from self play ===")
    self_play_generator = SelfPlayGenerator(model, args)
    position_generator = self_play_generator.position_generator()
    for file_idx in range(args.data_range_min, args.data_range_max):
        all_boards_tensor = torch.Tensor()
        all_mcts_policies = torch.Tensor()
        all_results = torch.Tensor()
        for _ in range(args.samples_per_file):
            board_tensor, mcts_policy, result = next(position_generator)
            all_boards_tensor = torch.cat((all_boards_tensor, board_tensor.unsqueeze(0)))
            all_mcts_policies = torch.cat((all_mcts_policies, mcts_policy.unsqueeze(0)))
            all_results = torch.cat((all_results, result.unsqueeze(0)))

        file_name = f'data/{args.run_name}_{file_idx}.pt'
        torch.save((all_boards_tensor, all_mcts_policies, all_results), file_name)
        logger.info(f'self-play data generation wrote {file_name}')


if __name__ == '__main__':
    main()
