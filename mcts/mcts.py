#!/usr/bin/env python3
import numpy as np
import torch
import device
from hexboard import Board, to_move
from hexconvolution import MCTSModel
from utils import dotdict
from logger import logger

class Node:
    def __init__(self, parent, move_idx):
        self._board = None # will be set upon first call of board(). For root node this needs to be set manually
        self.N = 0
        self.W = 0
        self.P = None
        self.children = []
        self.parent = parent
        self.move_idx = move_idx # last move on the board, None for root node

    @staticmethod
    def create_root_node(board):
        root = Node(parent=None, move_idx=None)
        root._board = board
        return root

    def board(self):
        if not self._board:
            self._board = self.parent.board().set_stone_immutable(to_move(self.move_idx, self.parent.board().size))
        return self._board

    def has_winner(self):
        return self.board().winner != False

    def is_leaf(self):
        return self.children == []

    def expand(self, model_policy):
        self.P = torch.exp(model_policy[0])
        for move_idx, p in enumerate(self.P):
            move = to_move(move_idx, self.board().size)
            if move in self.board().legal_moves and p > 1e-6:
                self.children.append(Node(self, move_idx))

    def backup(self, val):
        self.N += 1
        self.W += val
        if self.parent is not None:
            self.parent.backup(-val)

class MCTS:
    def __init__(self, model, root_board, args):
        self.model = model.to(device.device)
        self.args = args
        self.root = Node.create_root_node(board=root_board)

    def search(self):
        self._visit(self.root)

    def _visit(self, node: Node):
        if node.has_winner():
            node.backup(1) # player has won the game, last move was good
            return

        if node.is_leaf():
            self.expand_leaf(node)
            return

        best_child = self.find_best_move(node)

        self._visit(best_child)

    def find_best_move(self, node):
        max_val, best_child = -float("inf"), None

        for child in node.children:
            if node.N == 1:
                # formula from alpha go paper doesn't make much sense for this case.
                # just use model predicted policy here
                U = self.args.c_puct * node.P[child.move_idx]
                Q = 0
            else:
                U = self.args.c_puct * node.P[child.move_idx] * np.sqrt(node.N - 1) / (1 + child.N)
                # assert(node.N - 1 == sum(child.N for child in node.children))
                Q = child.W / child.N if child.N >= 1 else 0
            # logger.debug(f'{child.move_idx:2d} {Q:.3f} {U:.3f}')
            if U + Q > max_val:
                max_val = U + Q
                best_child = child

        return best_child

    def expand_leaf(self, node):
        with torch.no_grad():
            model_policy, model_value = self.model(node.board().board_tensor.unsqueeze(0).to(device.device))
            node.expand(model_policy)

        v = model_value[0]
        node.backup(v)

class MCTSSearch:
    def __init__(self, model, args):
        self.model = model
        self.args = args

    def move_counts(self, board):
        mcts = MCTS(model=self.model, root_board=board, args=self.args)
        for i in range(1, self.args.num_mcts_simulations + 1):
            mcts.search()
        counts = [0] * board.size * board.size
        Qs = [0] * board.size * board.size
        for child in mcts.root.children:
            counts[child.move_idx] = child.N
            Qs[child.move_idx] = child.W / child.N if child.N > 0 else 0

        logger.debug(torch.Tensor(Qs).view(5,5))
        return counts

    @staticmethod
    def move_probabilities(counts, temperature):
        if temperature < 1e-8:
            best_move = np.argmax(counts)
            probs = [0] * len(counts)
            logger.debug(f'probs = {probs}, best_move = {best_move}')
            probs[best_move] = 1
            return probs
        else:
            counts = [x ** (1. / temperature) for x in counts]
            probs = [x / float(sum(counts)) for x in counts]
            return probs


def test():
    board_size = 5
    model = MCTSModel(board_size=5, layers=2, intermediate_channels=1)

    args = dotdict({
        'c_puct': 1.,
        'num_mcts_simulations': 400
    })

    mcts_search = MCTSSearch(model, args)

    board = Board(board_size)
    probs = mcts_search.move_counts(board, temperature=1)
    logger.debug(torch.Tensor(probs).view(5,5))
    # TODO add noise

if __name__ == '__main__':
    # import cProfile
    # cProfile.run('test()', 'mcts.profile')
    test()
