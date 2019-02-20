#!/usr/bin/env python
import torch

from hexboard import Board
from hexgame import MultiHexGame

import argparse
from configparser import ConfigParser

from visualization.image import draw_board_image
from time import gmtime, strftime

def play_games(models, number_of_games, device, batch_size, temperature, board_size, plot_board):
    '''
    two models play against each other
    '''
    time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    result = [[0, 0], [0, 0]]
    for model_idx in range(2):        
        game_number = 0
        for batch_number in range(number_of_games // (2*batch_size)):
            ordered_models = models if model_idx == 0 else models[::-1]
            boards = [Board(size=board_size) for idx in range(batch_size)]
            multihexgame = MultiHexGame(boards=boards, models=ordered_models, device=device, noise=None, noise_parameters=None, temperature=temperature)
            multihexgame.play_moves()
            for board in multihexgame.boards:
                winning_model = board.winner[0]
                result[model_idx][winning_model] += 1
                if plot_board:
                    draw_board_image(board.board_tensor,
                        f'images/{time}_{model_idx}_{game_number:04d}.png')
                game_number += 1
        print(f'{result[model_idx][0+1*model_idx]} : {result[model_idx][1-1*model_idx]}')
    adbc = (result[0][0]*result[1][1] - result[0][1]*result[1][0])
    signed_chi_squared = 4*adbc*abs(adbc)/(number_of_games*(result[0][0]+result[1][0])*(result[0][1]+result[1][1])+1)
    print(f'signed_chi_squared = {signed_chi_squared}')
    return result, signed_chi_squared


def get_args(config_file):
    config = ConfigParser()
    config.read(config_file)
    parser = argparse.ArgumentParser()

    parser.add_argument('--model1', type=str, default=config.get('EVALUATE MODELS', 'model1'))
    parser.add_argument('--model2', type=str, default=config.get('EVALUATE MODELS', 'model2'))
    parser.add_argument('--number_of_games', type=int, default=config.getint('EVALUATE MODELS', 'number_of_games'))
    parser.add_argument('--batch_size', type=int, default=config.getint('EVALUATE MODELS', 'batch_size'))
    parser.add_argument('--board_size', type=int, default=config.getint('EVALUATE MODELS', 'board_size'))
    parser.add_argument('--temperature', type=float, default=config.getfloat('EVALUATE MODELS', 'temperature'))
    parser.add_argument('--plot_board', type=bool, default=config.getboolean('EVALUATE MODELS', 'plot_board'))

    return parser.parse_args()

def evaluate(config_file = 'config.ini'):
    print("=== evaluate two models ===")
    args = get_args(config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model1 = torch.load('models/{}.pt'.format(args.model1), map_location=device)
    model2 = torch.load('models/{}.pt'.format(args.model2), map_location=device)

    play_games((model1, model2), args.number_of_games, device, args.batch_size, args.temperature, args.board_size, args.plot_board)

if __name__ == '__main__':
    evaluate()