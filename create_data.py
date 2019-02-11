import torch
from torch.utils.data.dataset import Dataset
from torch.distributions.dirichlet import Dirichlet

import os
import argparse
from configparser import ConfigParser

from hexboard import Board
from hexgame import HexGame
from hexconvolution import NoMCTSModel


def generate_data_one_file(number_of_samples, model, device, run_name, noise, noise_level=0, temperature=1, board_size=11):
    position_idx = 0
    all_board_states = torch.Tensor()
    all_moves = torch.LongTensor()
    all_targets = torch.Tensor()
    try:
        os.makedirs("data")
    except:
        pass
    while position_idx < number_of_samples:
        board = Board(size=board_size)
        hexgame = HexGame(board, model, device, noise, noise_level, temperature)
        print('played_game')
        board_states, moves, targets = hexgame.play_moves()
        position_idx += len(board_states)
        all_board_states = torch.cat((all_board_states,board_states))
        all_moves = torch.cat((all_moves,moves))
        all_targets = torch.cat((all_targets,targets))
        print('created_game_data')
    torch.save((all_board_states, all_moves, all_targets),'data/{}.pt'.format(run_name))


config = ConfigParser()
config.read('config.ini')
parser = argparse.ArgumentParser()

parser.add_argument('--board_size', type=int, default=config.get('CREATE DATA', 'board_size'))
parser.add_argument('--model', type=str, default=config.get('CREATE DATA', 'model'))
parser.add_argument('--positions_count', type=int, default=config.get('CREATE DATA', 'positions_count'))
parser.add_argument('--run_name', type=str, default=config.get('CREATE DATA', 'run_name'))
parser.add_argument('--noise_alpha', type=float, default=config.get('CREATE DATA', 'noise_alpha'))
parser.add_argument('--noise_level', type=float, default=config.get('CREATE DATA', 'noise_level'))
parser.add_argument('--temperature', type=float, default=config.get('CREATE DATA', 'temperature'))

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('models/{}.pt'.format(args.model))
noise = Dirichlet(torch.full((args.board_size**2,), args.noise_alpha))

generate_data_one_file(args.positions_count, model, device, args.run_name, noise, args.noise_level, args.temperature, args.board_size)