#!/usr/bin/env python
import random
from configparser import ConfigParser
from time import gmtime, strftime

from hex.logic import hexboard
from hex.logic.hexgame import MultiHexGame
from hex.utils.logger import logger
from hex.utils.utils import load_model
from hex.visualization.image import draw_board_image


def play_games(models, num_opened_moves, number_of_games, batch_size, temperature, temperature_decay, plot_board, verbose=False):
    assert(len(models) == 2)
    assert(models[0].board_size == models[1].board_size)
    board_size = models[0].board_size

    if num_opened_moves > 0:
        openings = list(hexboard.first_k_moves(board_size, num_opened_moves))
        random.shuffle(openings)
        number_of_games = min(len(openings), number_of_games)

    logger.debug(f'playing {number_of_games} games')

    time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    result = [[0, 0], [0, 0]]

    logger.debug(f'    M1 - M2')
    for starting_model in range(2):
        game_number = 0

        while game_number < number_of_games:
            ordered_models = models[::-1] if starting_model^(num_opened_moves%2) else models
            if num_opened_moves > 0:
                batch_of_openings = openings[game_number:game_number + batch_size]
                boards = [hexboard.get_opened_board(board_size, opening) for opening in batch_of_openings]
            else:
                boards = [hexboard.Board(size=board_size) for idx in range(batch_size)]
            multihexgame = MultiHexGame(
                    boards,
                    ordered_models,
                    noise=None,
                    noise_parameters=None,
                    temperature=temperature,
                    temperature_decay=temperature_decay,
            )
            multihexgame.play_moves()
            for board in multihexgame.boards:
                winning_model = board.winner[0] if starting_model == 0 else 1 - board.winner[0]
                result[starting_model][winning_model] += 1
                if plot_board:
                    draw_board_image(board.board_tensor,
                        f'data/images/{time}_{starting_model}_{game_number:04d}.png')
                    board.export_as_FF4(f'images/{time}_{starting_model}_{game_number:04d}.txt')
                game_number += 1
        color_model1 = 'B' if starting_model == 0 else 'W'
        color_model2 = 'W' if starting_model == 0 else 'B'
        if verbose:
            logger.info(f'{color_model1}:{color_model2} {result[starting_model][0]} : {result[starting_model][1]}')
        else:
            logger.debug(f'{color_model1}:{color_model2} {result[starting_model][0]} : {result[starting_model][1]}')

    adbc = (result[0][0]*result[1][0] - result[0][1]*result[1][1])
    signed_chi_squared = 4*adbc*abs(adbc)/((result[0][0]+result[1][1]+result[0][1]+result[1][0])*\
        (result[0][0]+result[1][1])*(result[0][1]+result[1][0])+1)
    logger.debug(f'signed_chi_squared = {signed_chi_squared}')
    return result, signed_chi_squared


def evaluate(config_file):
    logger.info("")
    logger.info("=== evaluating two models ===")
    config = ConfigParser()
    config.read(config_file)

    model1 = load_model(f"models/{config.get('EVALUATE MODELS', 'model1')}.pt")
    model2 = load_model(f"models/{config.get('EVALUATE MODELS', 'model2')}.pt")

    play_games(
            models=(model1, model2),
            num_opened_moves=config.getint('EVALUATE MODELS', 'num_opened_moves'),
            number_of_games=config.getint('EVALUATE MODELS', 'number_of_games'),
            batch_size=config.getint('EVALUATE MODELS', 'batch_size'),
            temperature=config.getfloat('EVALUATE MODELS', 'temperature'),
            temperature_decay=config.getfloat('EVALUATE MODELS', 'temperature_decay'),
            plot_board=config.getboolean('EVALUATE MODELS', 'plot_board'),
            verbose=True
        )


if __name__ == '__main__':
    evaluate('config.ini')
