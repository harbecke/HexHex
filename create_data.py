import torch
from torch.utils.data.dataset import Dataset
from torch.distributions.dirichlet import Dirichlet

import os
import argparse
from configparser import ConfigParser

from hexboard import Board
from hexgame import HexGame
from hexconvolution import NoMCTSModel


def generate_data_one_file(number_of_samples, board_size, model, device, run_name):
    position_idx = 0
    all_board_states = torch.Tensor()
    all_moves = torch.LongTensor()
    all_targets = torch.LongTensor()
    try:
    	os.makedirs("data")
    	os.makedirs('models')
    except:
    	pass
    while position_idx < number_of_samples:
        board = Board(size=board_size)
        hexgame = HexGame(board, model, device)
        board_states, moves, targets = hexgame.play_moves()
        position_idx += len(board_states)
        all_board_states = torch.cat((all_board_states,board_states))
        all_moves = torch.cat((all_moves,moves))
        all_targets = torch.cat((all_targets,targets))
    torch.save((all_board_states, all_moves, all_targets),'data/{}.pt'.format(run_name))
    torch.save(model, 'models/{}.pt'.format(run_name))


config = ConfigParser()
config.read('config.INI')
parser = argparse.ArgumentParser()

parser.add_argument('--board_size', type=int, default=config.get('CREATE DATA', 'board_size'))
parser.add_argument('--model_layer', type=int, default=config.get('CREATE DATA', 'model_layer'))
parser.add_argument('--positions_count', type=int, default=config.get('CREATE DATA', 'positions_count'))

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noise = Dirichlet(torch.full((args.board_size**2,),1))
model = NoMCTSModel(board_size=args.board_size, layers=args.model_layer, noise=noise)

generate_data_one_file(args.positions_count, args.board_size, model, device, 'first_test')