import numpy as np
import torch

from hex.logic.hexboard import Board
from hex.utils import utils


class Node:
    """Monte Carlo Tree Search node"""
    def __init__(self, board, config, model):
        self.board = board
        self.Q = [utils.Average() for _ in range(board.size ** 2)]
        self.children = [None] * board.size ** 2
        self.config = config
        self.model = model
        with torch.no_grad():
            self.model_output = self.model(self.board.board_tensor.unsqueeze(0))[0]
        for idx, q in enumerate(self.Q):
            model_output = self.model_output[idx]
            if model_output > -900:
                q.add(torch.sigmoid(model_output), 1)

    def visit(self):
        if self.board.winner:
            # cannot chose another move as the opponent has already won
            return 0

        move = self.choose_move()
        visit_count = self.Q[move].num_samples

        if visit_count == 0:
            #  the child node has not been chosen before
            #  -> propagate evaluation up
            value = torch.sigmoid(self.model_output[move]).item()
        else:
            # child node has been chosen before -> go deeper
            if visit_count == 1:
                # child node has been chosen but not yet expanded and evaluated
                child = Node(self.board.set_stone_immutable(move), self.config, self.model)
                self.children[move] = child
            value = 1 - self.children[move].visit()

        self.Q[move].add(value, 1)
        return value

    def choose_move(self):
        Q = torch.tensor([q.mean() for q in self.Q]).type(torch.float)
        N = torch.tensor([q.num_samples for q in self.Q]).type(torch.float)
        c_puct = self.config.getfloat('c_puct', 1.25)
        U = c_puct * torch.sigmoid(self.model_output) * N.sum().sqrt() / (1 + N)
        legal_mask = (self.model_output > -900).type(torch.float)
        total = ((Q + U)*legal_mask)
        move = total.argmax()
        return move.item()


class Simulation:
    def __init__(self, model, config, board: Board):
        self.model = model
        self.config = config
        self.root = Node(board, config, model)

    def run(self):
        for _ in range(self.config.getint('num_mcts_simulations', 10)):
            self.run_simulation()

        return [q.num_samples for q in self.root.Q]

    def run_simulation(self):
        self.root.visit()


class Game:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def single_move(self, board):
        simulation = Simulation(self.model, self.config, board)
        move_counts = simulation.run()
        move = np.argmax(move_counts)
        board.set_stone(move.item())
        return move_counts
