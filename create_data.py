import torch
from torch.utils.data.dataset import Dataset
from torch.distributions.dirichlet import Dirichlet

import os
import argparse
from configparser import ConfigParser

from hexboard import Board
from hexgame import HexGame
from hexconvolution import NoMCTSModel


def generate_data_files(number_of_files, samples_per_file, model, device, run_name, noise, noise_level=0, temperature=1, board_size=11):
    all_board_states = torch.Tensor()
    all_moves = torch.LongTensor()
    all_targets = torch.Tensor()

    file_counter = 0
    while file_counter < number_of_files:
        while all_board_states.shape[0] < samples_per_file:
            
            board = Board(size=board_size)
            hexgame = HexGame(board, model, device, noise, noise_level, temperature)
            board_states, moves, targets, _ = hexgame.play_moves()

            all_board_states = torch.cat((all_board_states,board_states))
            all_moves = torch.cat((all_moves,moves))
            all_targets = torch.cat((all_targets,targets))
        
        file_name = f'data/{run_name}_{file_counter}.pt'
        torch.save((all_board_states[:samples_per_file], all_moves[:samples_per_file], all_targets[:samples_per_file]), file_name)
        print(f'written_{file_name}')
        file_counter += 1
        
        all_board_states = all_board_states[samples_per_file:]
        all_moves = all_moves[samples_per_file:]
        all_targets = all_targets[samples_per_file:]


config = ConfigParser()
config.read('config.ini')
parser = argparse.ArgumentParser()

parser.add_argument('--number_of_files', type=int, default=config.get('CREATE DATA', 'number_of_files'))
parser.add_argument('--samples_per_file', type=int, default=config.get('CREATE DATA', 'samples_per_file'))
parser.add_argument('--model', type=str, default=config.get('CREATE DATA', 'model'))
parser.add_argument('--run_name', type=str, default=config.get('CREATE DATA', 'run_name'))
parser.add_argument('--noise_alpha', type=float, default=config.get('CREATE DATA', 'noise_alpha'))
parser.add_argument('--noise_level', type=float, default=config.get('CREATE DATA', 'noise_level'))
parser.add_argument('--temperature', type=float, default=config.get('CREATE DATA', 'temperature'))
parser.add_argument('--board_size', type=int, default=config.get('CREATE DATA', 'board_size'))


args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('models/{}.pt'.format(args.model), map_location=device)
noise = Dirichlet(torch.full((args.board_size**2,), args.noise_alpha))

generate_data_files(args.number_of_files, args.samples_per_file, model, device, args.run_name, noise, args.noise_level, args.temperature, args.board_size)