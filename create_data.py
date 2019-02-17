import torch
from torch.utils.data.dataset import Dataset
from torch.distributions.dirichlet import Dirichlet

import os
import argparse
from configparser import ConfigParser

from hexboard import Board
from hexgame import MultiHexGame
from hexconvolution import NoMCTSModel


def generate_data_files(number_of_files, samples_per_file, model, device, batch_size, run_name, noise_level=0, noise_alpha=0.03, temperature=1, board_size=11):
    all_board_states = torch.Tensor()
    all_moves = torch.LongTensor()
    all_targets = torch.Tensor()

    file_counter = 0
    while file_counter < number_of_files:
        while all_board_states.shape[0] < samples_per_file:
            boards = [Board(size=board_size) for idx in range(batch_size)]
            multihexgame = MultiHexGame(boards, (model,), device, noise_level, noise_alpha, temperature)
            board_states, moves, targets = multihexgame.play_moves()

            all_board_states = torch.cat((all_board_states,board_states))
            all_moves = torch.cat((all_moves,moves))
            all_targets = torch.cat((all_targets,targets))
        
        file_name = f'data/{run_name}_{file_counter}.pt'
        torch.save((all_board_states[:samples_per_file], all_moves[:samples_per_file], all_targets[:samples_per_file]), file_name)
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

    parser.add_argument('--number_of_files', type=int, default=config.get('CREATE DATA', 'number_of_files'))
    parser.add_argument('--samples_per_file', type=int, default=config.get('CREATE DATA', 'samples_per_file'))
    parser.add_argument('--model', type=str, default=config.get('CREATE DATA', 'model'))
    parser.add_argument('--batch_size', type=int, default=config.get('CREATE DATA', 'batch_size'))
    parser.add_argument('--run_name', type=str, default=config.get('CREATE DATA', 'run_name'))
    parser.add_argument('--noise_level', type=float, default=config.get('CREATE DATA', 'noise_level'))
    parser.add_argument('--noise_alpha', type=float, default=config.get('CREATE DATA', 'noise_alpha'))
    parser.add_argument('--temperature', type=float, default=config.get('CREATE DATA', 'temperature'))
    parser.add_argument('--board_size', type=int, default=config.get('CREATE DATA', 'board_size'))

    return parser.parse_args()

def main(config_file = 'config.ini'):
    print("=== creating data from self play ===")
    args = get_args(config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('models/{}.pt'.format(args.model), map_location=device)

    generate_data_files(args.number_of_files, args.samples_per_file, model, device, args.batch_size, args.run_name, args.noise_level, args.noise_alpha, args.temperature, args.board_size)

if __name__ == '__main__':
    main()