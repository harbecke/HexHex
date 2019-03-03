#!/usr/bin/env python3
import numpy as np
from configparser import ConfigParser
import argparse
import torch

from logger import logger
from hexboard import Board
from hexgame import MultiHexGame
from mcts.mcts import MCTSSearch
from hexboard import to_move
from visualization.gui import Gui

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
    parser.add_argument('--num_mcts_simulations', type=int, default=config.getint('INTERACTIVE', 'num_mcts_simulations'))
    parser.add_argument('--mcts_batch_size', type=int, default=config.getint('INTERACTIVE', 'mcts_batch_size'))
    parser.add_argument('--c_puct', type=float, default=config.getint('INTERACTIVE', 'c_puct'))
    parser.add_argument('--n_virtual_loss', type=int, default=config.getint('INTERACTIVE', 'n_virtual_loss'))

    return parser.parse_args()

class InteractiveGame:
    '''
    allows to play a game against a model
    '''
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load('models/{}.pt'.format(args.model), map_location=self.device)
        self.board = Board(size=self.model.board_size)
        self.gui = Gui(self.board, args.gui_radius)
        self.mcts_search = MCTSSearch(self.model, args)
        self.args = args
        self.model_type = args.model_type
        self.game = MultiHexGame(boards=(self.board,), models=(self.model,), device=self.device, noise=None, noise_parameters=None, temperature=args.temperature, temperature_decay=args.temperature_decay) # only for NoMCTS arch

    def play_human_move(self):
        move = self.get_move()
        self.board.set_stone(move)
        self.gui.update_board(self.board)

        if self.board.winner:
            logger.info("Player has won!")
            self.wait_for_gui_exit()

    def play_ai_move(self):
        if self.args.model_type == 'mcts':
            move_counts, Qs = self.mcts_search.simulate(self.board)
            move_ratings = self.mcts_search.move_probabilities(move_counts, self.args.temperature)
            move_idx = self.mcts_search.sample_move(move_ratings)
            print(f'Expected outcome for agent: {Qs[move_idx]}')
            self.board.set_stone(to_move(move_idx, self.board.size))
            self.gui.update_board(self.board, field_text=np.array(move_counts))

        else:
            move_ratings = self.game.batched_single_move(self.model)
            field_text = [f'{int(100*rating)}' for rating in move_ratings[0]]
            self.gui.update_board(self.board, field_text=field_text)

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
