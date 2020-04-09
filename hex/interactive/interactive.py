#!/usr/bin/env python3
from configparser import ConfigParser

import numpy as np

from hex.interactive.gui import Gui
from hex.logic.hexboard import Board
from hex.logic.hexgame import MultiHexGame
from hex.model import mcts
from hex.utils.logger import logger
from hex.utils.utils import load_model


class InteractiveGame:
    """
    allows to play a game against a model
    """

    def __init__(self, config):
        self.config = config['INTERACTIVE']
        self.model = load_model(f'models/{self.config.get("model", fallback="11_2w4_1100")}.pt')
        self.switch_allowed = self.config.getboolean('switch', fallback=True)
        self.board = Board(size=self.model.board_size, switch_allowed=self.switch_allowed)
        self.gui = Gui(self.board, self.config.getint('gui_radius', 50),
                       self.config.getboolean('dark_mode', False))
        if self.config.get('mode') == 'mcts':
            self.game = mcts.Game(self.model, config['INTERACTIVE'])
        else:
            self.game = MultiHexGame(
                boards=(self.board,),
                models=(self.model,),
                noise=None,
                noise_parameters=None,
                temperature=self.config.getfloat('temperature', fallback=0.0),
                temperature_decay=self.config.getfloat('temperature_decay', fallback=1.)
            )

    def print_ratings(self):
        ratings = self.model(self.board.board_tensor.unsqueeze(0)).view(self.board.size, self.board.size)
        if self.board.player:
            ratings = ratings.transpose(0, 1)
        with np.printoptions(precision=1, suppress=True):
            logger.info('I politely recommend the following ratings\n' + str(ratings.detach().numpy()))

    def play_move(self):
        self.print_ratings()
        move = self.get_move()
        if move == 'show_ratings':
            self.gui.show_field_text = not self.gui.show_field_text
        elif move == 'redraw':
            pass
        elif move == 'ai_move':
            self.play_ai_move()
        elif move == 'undo_move':
            self.undo_move()
        else:
            self.board.set_stone(move)
            if self.board.winner:
                logger.info("Player has won")
            elif not self.gui.editor_mode:
                self.print_ratings()
                self.play_ai_move()
        self.gui.update_board(self.board)

    def undo_move(self):
        self.board.undo_move_board()
        # forgot know what the ratings were better not show anything
        self.gui.last_field_text = None
        self.gui.update_board(self.board)

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
        self.gui.update_field_text(rating_strings)
        self.gui.update_board(self.board)

        if self.board.winner:
            logger.info("agent has won!")

    def get_move(self):
        while True:
            move = self.gui.get_move()
            if move in ['ai_move', 'undo_move', 'redraw', 'show_ratings']:
                return move
            if move in self.board.legal_moves:
                return move


def play_game(interactive):
    while True:
        interactive.play_move()
        if interactive.board.winner:
            break


def _main():
    logger.info("Starting interactive game")

    config = ConfigParser()
    config.read('config.ini')

    while True:
        interactive = InteractiveGame(config)
        play_game(interactive)
        interactive.gui.wait_for_click()  # wait for click to start new game


if __name__ == '__main__':
    _main()
