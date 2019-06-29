#!/usr/bin/env python
import argparse
from configparser import ConfigParser
from time import gmtime, strftime

import torch

from hex.logic import hexboard
from hex.logic.hexboard import Board
from hex.logic.hexgame import MultiHexGame
from hex.utils.utils import load_model
from hex.visualization.image import draw_board_image


def play_games(models, device, openings, number_of_games, batch_size, temperature, temperature_decay, plot_board):
    assert(len(models) == 2)
    assert(models[0].board_size == models[1].board_size)
    board_size = models[0].board_size
    if openings:
        openings = list(hexboard.first_k_moves(board_size, 2))
        number_of_games = len(openings)

    print(f'playing {number_of_games} games')

    time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    result = [[0, 0], [0, 0]]

    print(f'    M1 - M2')
    for starting_model in range(2):
        game_number = 0

        while game_number < number_of_games:
            ordered_models = models if starting_model == 0 else models[::-1]
            if openings:
                batch_of_openings = openings[game_number:game_number + batch_size]
                boards = [hexboard.get_opened_board(board_size, opening) for opening in batch_of_openings]
            else:
                boards = [Board(size=board_size) for idx in range(batch_size)]
            multihexgame = MultiHexGame(
                    boards,
                    ordered_models,
                    noise=None,
                    noise_parameters=None,
                    device=device,
                    temperature=temperature,
                    temperature_decay=temperature_decay,
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
        color_model1 = 'B' if starting_model == 0 else 'W'
        color_model2 = 'W' if starting_model == 0 else 'B'
        print(f'{color_model1}:{color_model2} {result[starting_model][0]} : {result[starting_model][1]}')

    adbc = (result[0][0]*result[1][0] - result[0][1]*result[1][1])
    signed_chi_squared = 4*adbc*abs(adbc)/((result[0][0]+result[1][1]+result[0][1]+result[1][0])*(result[0][0]+result[1][1])*(result[0][1]+result[1][0])+1)
    print(f'signed_chi_squared = {signed_chi_squared}')
    return result, signed_chi_squared


def get_args(config_file):
    config = ConfigParser()
    config.read(config_file)
    parser = argparse.ArgumentParser()

    parser.add_argument('--model1', type=str, default=config.get('EVALUATE MODELS', 'model1'))
    parser.add_argument('--model2', type=str, default=config.get('EVALUATE MODELS', 'model2'))
    parser.add_argument('--openings', type=bool, default=config.getboolean('EVALUATE MODELS', 'openings'))
    parser.add_argument('--number_of_games', type=int, default=config.getint('EVALUATE MODELS', 'number_of_games'))
    parser.add_argument('--batch_size', type=int, default=config.getint('EVALUATE MODELS', 'batch_size'))
    parser.add_argument('--board_size', type=int, default=config.getint('EVALUATE MODELS', 'board_size'))
    parser.add_argument('--temperature', type=float, default=config.getfloat('EVALUATE MODELS', 'temperature'))
    parser.add_argument('--temperature_decay', type=float, default=config.getfloat('EVALUATE MODELS', 'temperature_decay'))
    parser.add_argument('--plot_board', type=bool, default=config.getboolean('EVALUATE MODELS', 'plot_board'))

    return parser.parse_args(args=[])


def evaluate(config_file = 'config.ini'):
    print("=== evaluate two models ===")
    args = get_args(config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model1, _ = load_model(f'models/{args.model1}.pt')
    model2, _ = load_model(f'models/{args.model2}.pt')

    play_games(
            models=(model1, model2),
            openings=args.openings,
            number_of_games=args.number_of_games,
            device=device,
            batch_size=args.batch_size,
            board_size=args.board_size,
            temperature=args.temperature,
            temperature_decay=args.temperature_decay,
            plot_board=args.plot_board
        )


if __name__ == '__main__':
    evaluate()
