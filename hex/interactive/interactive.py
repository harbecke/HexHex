#!/usr/bin/env python3
import argparse
from configparser import ConfigParser

import torch

from hex.interactive.gui import Gui
from hex.logic.hexboard import Board
from hex.logic.hexgame import MultiHexGame
from hex.utils.logger import logger
from hex.utils.utils import load_model


def get_args():
    config = ConfigParser()
    config.read('config.ini')
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default=config.get('INTERACTIVE', 'model'))
    parser.add_argument('--model_type', type=str, default=config.get('INTERACTIVE', 'model_type'))
    parser.add_argument('--temperature', type=float, default=config.getfloat('INTERACTIVE', 'temperature'))
    parser.add_argument('--temperature_decay', type=float, default=config.getfloat('INTERACTIVE', 'temperature_decay'))
    parser.add_argument('--first_move_ai', type=bool, default=config.getboolean('INTERACTIVE', 'first_move_ai'))
    parser.add_argument('--gui_radius', type=int, default=config.getint('INTERACTIVE', 'gui_radius'))
    parser.add_argument('--noise_epsilon', type=float, default=config.getfloat('INTERACTIVE', 'noise_epsilon'))
    parser.add_argument('--noise_spread', type=float, default=config.getfloat('INTERACTIVE', 'noise_spread'))

    return parser.parse_args()

class InteractiveGame:
    '''
    allows to play a game against a model
    '''
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(f'models/{args.model}.pt')
        self.board = Board(size=self.model.board_size)
        self.gui = Gui(self.board, args.gui_radius)
        self.args = args
        self.model_type = args.model_type
        self.game = MultiHexGame(boards=(self.board,), models=(self.model,), device=self.device, noise=None, 
            noise_parameters=None, temperature=args.temperature, temperature_decay=args.temperature_decay)

    def play_human_move(self):
        move = self.get_move()
        self.board.set_stone(move)
        self.gui.update_board(self.board)

        if self.board.winner:
            logger.info("Player has won!")
            self.wait_for_gui_exit()

    def play_ai_move(self):
        move_ratings = self.game.batched_single_move(self.model)
        rating_strings = []
        for rating in move_ratings[0]:
            if rating > 99:
                rating_strings.append('+')
            elif rating < -99:
                rating_strings.append('-')
            else:
                rating_strings.append("{0:0.1f}".format(rating))
        #field_text = ["{0:0.1f}".format(rating) for rating in move_ratings[0]]
        self.gui.update_board(self.board, field_text=rating_strings)

        if self.board.winner:
            logger.info("agent has won!")
            self.wait_for_gui_exit()

    def wait_for_gui_exit(self):
        while True:
            self.gui.get_cell()

    def get_move(self):
        while True:
            move = self.gui.get_cell()
            if move in self.board.legal_moves:
                return move


def _main():
    logger.info("Starting interactive game")

    args = get_args()
    interactive = InteractiveGame(args)

    if args.first_move_ai:
        interactive.play_ai_move()
    while True:
        interactive.play_human_move()
        interactive.play_ai_move()


if __name__ == '__main__':
    _main()
