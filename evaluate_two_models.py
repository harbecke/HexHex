import torch

from hexboard import Board
from hexgame import HexGameTwoModels

import argparse
from configparser import ConfigParser

from visualization.image import draw_board_image
import time

def play_games(model1, model2, number_of_games, device, temperature, board_size, draw_game):
    result = [0, 0]

    for _ in range(number_of_games // 2):
        for game_idx in range(2):
            first_model = model1 if game_idx == 0 else model2
            second_model = model2 if game_idx == 0 else model1
            board = Board(size=board_size)
            game = HexGameTwoModels(board, first_model, second_model, device, temperature)
            winner = game.play_moves()
            if draw_game:
                draw_board_image(board.board_tensor, f'images/{time.time()}.png')
            winning_model = winner if game_idx == 0 else 1 - winner
            result[winning_model] += 1
        print(result)
    return result


config = ConfigParser()
config.read('config.ini')
parser = argparse.ArgumentParser()

parser.add_argument('--model1', type=str, default=config.get('EVALUATE MODELS', 'model1'))
parser.add_argument('--model2', type=str, default=config.get('EVALUATE MODELS', 'model2'))
parser.add_argument('--number_of_games', type=int, default=config.get('EVALUATE MODELS', 'number_of_games'))
parser.add_argument('--board_size', type=int, default=config.get('EVALUATE MODELS', 'board_size'))
parser.add_argument('--temperature', type=float, default=config.get('EVALUATE MODELS', 'temperature'))
parser.add_argument('--draw_game', type=bool, default=config.getboolean('EVALUATE MODELS', 'draw_game'))

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model1 = torch.load('models/{}.pt'.format(args.model1), map_location=device)
model2 = torch.load('models/{}.pt'.format(args.model2), map_location=device)

play_games(model1, model2, args.number_of_games, device, args.temperature, args.board_size, args.draw_game)