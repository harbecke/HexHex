import torch

from hexboard import Board
from hexgame import HexGameTwoModels

import argparse
from configparser import ConfigParser

def play_games(model1, model2, number_of_games, device, temperature, board_size):
    result = [0, 0]
    for idx in range(number_of_games):
        board = Board(size=board_size)
        game = HexGameTwoModels(board, model1, model2, device, temperature)
        winner = game.play_moves()
        result[winner] += 1
    print(result)
    for idx in range(number_of_games):
        board = Board(size=board_size)
        game = HexGameTwoModels(board, model2, model1, device, temperature)
        winner = game.play_moves()
        result[1-winner] += 1
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

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model1 = torch.load('models/{}.pt'.format(args.model1), map_location=device)
model2 = torch.load('models/{}.pt'.format(args.model2), map_location=device)

play_games(model1, model2, args.number_of_games, device, args.temperature, args.board_size)