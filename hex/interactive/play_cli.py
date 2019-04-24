#!/usr/bin/env python3
import sys

import argparse
import logging
import torch
from configparser import ConfigParser

from hex.logic import hexboard
from hex.logic.hexgame import MultiHexGame
from hex.utils.utils import device, load_model

logging.basicConfig(level=logging.DEBUG, filename='play_cli.log', filemode='w')


def get_args():
    config = ConfigParser()
    config.read('config.ini')
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default=config.get('PLAY CLI', 'model'))
    parser.add_argument('--temperature', type=float, default=config.getfloat('PLAY CLI', 'temperature'))
    parser.add_argument('--temperature_decay', type=float, default=config.getfloat('PLAY CLI', 'temperature_decay'))

    return parser.parse_args()


class CliGame:
    def __init__(self, args):
        self.board = None
        self.device = device
        self.model, _ = load_model(f'models/{args.model}.pt')
        self.args = args


    def respond(self, line):
        splitted = line.split(' ')
        if splitted[0] == 'name':
            return 'HexHex'
        if splitted[0] == 'version':
            return '0.0'
        if splitted[0] == 'list_commands':
            return 'final_score'
        if splitted[0] == 'boardsize':
            self.board = hexboard.Board(int(splitted[1]))
            self.game = MultiHexGame(
                    boards=(self.board,),
                    models=(self.model,),
                    device=self.device,
                    noise=None,
                    noise_parameters=None,
                    temperature=self.args.temperature,
                    temperature_decay=self.args.temperature_decay
            )
            return ''
        if splitted[0] == 'showboard':
            return ''
        if splitted[0] == 'play':
            color = splitted[1]
            position = splitted[2]
            y = ord(position[0]) - ord('a')
            x = int(position[1:]) - 1
            logging.debug(f'interpreted move at {x}{y}')
            self.board.set_stone((x, y))
        if splitted[0] == 'genmove':
            if self.board.winner:
                return 'resign'
            self.game.batched_single_move(self.model)
            move = self.board.move_history[-1][1]
            alpha, numeric = hexboard.position_to_alpha_numeric(move)
            logging.debug(f'moving to {move}')
            return f'{alpha}{numeric}'
        if splitted[0] == 'final_score':
            winner = 'B' if self.board.winner[0] == 0 else 'W'
            return f'{winner}+'
        if splitted[0] == 'quit':
            exit(0)


def main():
    logging.info("Starting play_cli.py")
    args = get_args()
    game = CliGame(args)
    while True:
        logging.info(f'reading input')
        line = input()
        logging.info(f'input: {line}')
        answer = game.respond(line)
        logging.info(f'output: {answer}')
        print(f'= {answer}\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
