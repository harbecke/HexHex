#!/usr/bin/env python3
import argparse
from configparser import ConfigParser

import numpy as np

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
    parser.add_argument('--temperature', type=float, default=config.getfloat('INTERACTIVE', 'temperature'))
    parser.add_argument('--temperature_decay', type=float, default=config.getfloat('INTERACTIVE', 'temperature_decay'))
    parser.add_argument('--gui_radius', type=int, default=config.getint('INTERACTIVE', 'gui_radius'))
    parser.add_argument('--noise_epsilon', type=float, default=config.getfloat('INTERACTIVE', 'noise_epsilon'))
    parser.add_argument('--noise_spread', type=float, default=config.getfloat('INTERACTIVE', 'noise_spread'))

    return parser.parse_args()


class InteractiveGame:
    """
    allows to play a game against a model
    """
    def __init__(self, args):
        self.model = load_model(f'models/{args.model}.pt')
        self.board = Board(size=self.model.board_size)
        self.gui = Gui(self.board, args.gui_radius)
        self.args = args
        self.game = MultiHexGame(boards=(self.board,), models=(self.model,), noise=None,
            noise_parameters=None, temperature=args.temperature, temperature_decay=args.temperature_decay)

    def play_move(self):
        ratings = self.model(self.board.board_tensor.unsqueeze(0)).view(self.board.size, self.board.size)
        with np.printoptions(precision=1, suppress=True):
            logger.info('I politely recommend the following ratings\n' + str(ratings.detach().numpy()))
        move = self.get_move()
        if move == 'ai_move':
            self.play_ai_move()
        else:
            self.board.set_stone(move)
            self.gui.update_board(self.board)
            if self.board.winner:
                logger.info("Player has won")
            elif not self.gui.editor_mode:
                self.play_ai_move()

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
        self.gui.update_board(self.board, field_text=rating_strings)

        if self.board.winner:
            logger.info("agent has won!")

    def get_move(self):
        while True:
            move = self.gui.get_move()
            if move == 'ai_move':
                return move
            if move in self.board.legal_moves:
                return move


def _main():
    logger.info("Starting interactive game")
    logger.info("Press 'e' for editor mode")
    logger.info("Press 'a' to trigger ai move")

    args = get_args()
    while True:
        interactive = InteractiveGame(args)
        play_game(args, interactive)
        interactive.gui.wait_for_click()  # wait for click to start new game


def play_game(args, interactive):
    while True:
        interactive.play_move()
        if interactive.board.winner:
            break


if __name__ == '__main__':
    _main()
