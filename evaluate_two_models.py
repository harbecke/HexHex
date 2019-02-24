#!/usr/bin/env python
import torch

import hexboard
from hexboard import Board
from hexgame import MultiHexGame

import argparse
from configparser import ConfigParser

from visualization.image import draw_board_image
from time import gmtime, strftime


def play_all_openings(models, device, batch_size, board_size, plot_board):
    openings = list(hexboard.first_k_moves(board_size, 2))

    print(f'playing {len(openings)} openings')
    time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    result = [[0, 0], [0, 0]]

    number_of_games = 0

    print(f'    M1 - M2')
    for starting_model in range(2):
        game_number = 0

        while game_number < len(openings):
            ordered_models = models if starting_model == 0 else models[::-1]
            batch_of_openings = openings[game_number:game_number + batch_size]
            boards = [hexboard.get_opened_board(board_size, opening) for opening in batch_of_openings]
            multihexgame = MultiHexGame(
                    boards,
                    ordered_models,
                    noise=None,
                    noise_parameters=None,
                    device=device,
                    temperature=0.,
                    temperature_decay=0.,
            )
            multihexgame.play_moves()
            for board in multihexgame.boards:
                winning_model = board.winner[0] if starting_model == 0 else 1 - board.winner[0]
                result[starting_model][winning_model] += 1
                if plot_board:
                    draw_board_image(board.board_tensor,
                        f'images/{time}_{starting_model}_{game_number:04d}.png')
                    board.export_as_FF4(f'images/{time}_{starting_model}_{game_number:04d}.txt')
                game_number += 1
                number_of_games += 1
        color_model1 = 'B' if starting_model == 0 else 'W'
        color_model2 = 'W' if starting_model == 0 else 'B'
        print(f'{color_model1}:{color_model2} {result[starting_model][0]} : {result[starting_model][1]}')

    adbc = (result[0][0]*result[1][0] - result[0][1]*result[1][1])
    signed_chi_squared = 4*adbc*abs(adbc)/(number_of_games*(result[0][0]+result[1][1])*(result[0][1]+result[1][0])+1)
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
    parser.add_argument('--temperature_decay', type=float, default=config.getfloat('EVALUATE MODELS', 'temperature_decay'))
    parser.add_argument('--plot_board', type=bool, default=config.getboolean('EVALUATE MODELS', 'plot_board'))

    return parser.parse_args()

def evaluate(config_file = 'config.ini'):
    print("=== evaluate two models ===")
    args = get_args(config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model1 = torch.load('models/{}.pt'.format(args.model1), map_location=device)
    model2 = torch.load('models/{}.pt'.format(args.model2), map_location=device)

    play_all_openings(
            models=(model1, model2),
            device=device,
            batch_size=args.batch_size,
            board_size=args.board_size,
            plot_board=args.plot_board
    )

if __name__ == '__main__':
    evaluate()