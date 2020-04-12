#!/usr/bin/env python3
import logging
import sys
from configparser import ConfigParser

from hexhex.logic import hexboard
from hexhex.logic.hexgame import MultiHexGame
from hexhex.utils.utils import load_model

logging.basicConfig(level=logging.DEBUG, filename='play_cli.log', filemode='w')



class CliGame:
    def __init__(self, config):
        self.config = config['PLAY CLI']
        self.board = None
        self.switch = self.config.getboolean('switch', True)
        self.model = load_model(f'models/{self.config.get("model")}.pt')


    def respond(self, line):
        splitted = line.split(' ')
        if splitted[0] == 'name':
            return 'HexHex'
        if splitted[0] == 'version':
            return '0.0'
        if splitted[0] == 'list_commands':
            return 'final_score'
        if splitted[0] == 'boardsize':
            self.board = hexboard.Board(int(splitted[1]), self.switch)
            self.game = MultiHexGame(
                    boards=(self.board,),
                    models=(self.model,),
                    noise=None,
                    noise_parameters=None,
                    temperature=self.config.getfloat('temperature', 0.),
                    temperature_decay=self.config.getfloat('temperature_decay', 1.),
            )
            return ''
        if splitted[0] == 'showboard':
            return str(self.board.logical_board_tensor[0]-self.board.logical_board_tensor[1])
        if splitted[0] == 'play':
            color = splitted[1]
            position = splitted[2]
            if splitted[2] != 'resign':
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
    config = ConfigParser()
    config.read('config.ini')
    game = CliGame(config)
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
