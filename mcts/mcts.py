#!/usr/bin/env python3
import numpy as np
import torch
import device
from hexboard import Board

def to_move_idx(move, board_size):
    return move[0] * board_size + move[1]


def to_move(move_idx, board_size):
    return move_idx // board_size, move_idx % board_size


class Node:
    def __init__(self, board, parent, move_idx):
        self.board = board
        self.N = 0
        self.W = 0
        self.P = None
        self.children = []
        self.parent = parent
        self.move_idx = move_idx # last move on the board

    def has_winner(self):
        return self.board.winner != False

    def is_leaf(self):
        return self.children == []

    def expand(self, model_policy):
        self.P = model_policy
        for move_idx, p in enumerate(model_policy):
            move = to_move(move_idx, self.board.size)
            if move in self.board.legal_moves and p > 1e-8:
                new_board = self.board.set_stone_immutable(move)
                self.children.append(Node(new_board, self, move_idx))

    def backup(self, val):
        self.N += 1
        self.W += val
        if self.parent is not None:
            self.parent.backup(1 - val)

class MCTS:
    def __init__(self, model, c_puct, root_board, device):
        self.model = model.to(device)
        self.c_puct = c_puct  # exploration parameter
        self.root = Node(board=root_board, parent=None, move_idx=None)
        self.device = device

    def search(self):
        self._visit(self.root)

    def _visit(self, node: Node):
        """
        Taken from http://web.stanford.edu/~surag/posts/alphazero.html
        """

        if node.has_winner():
            node.backup(0) # previous player has won the game -> this player has lost
            return

        if node.is_leaf():
            with torch.no_grad():
                model_policy = self.model(node.board.board_tensor.unsqueeze(0).to(self.device))[0]
                node.expand(model_policy)

            v = .5  # TODO model should also predict rating for this position
            node.backup(v)
            return

        max_val, best_child = -float("inf"), None

        for child in node.children:
            if node.N == 1:
                # formula from alpha go paper doesn't make much sense for this case.
                # just use model predicted policy here
                U = self.c_puct * node.P[child.move_idx]
                Q = .5
            else:
                U = self.c_puct * node.P[child.move_idx] * np.sqrt(node.N - 1) / (1 + child.N)
                # assert(node.N - 1 == sum(child.N for child in node.children))
                Q = child.W / child.N if child.N >= 1 else .5
            # print(f'{child.move_idx:2d} {Q:.3f} {U:.3f}')
            if U + Q > max_val:
                max_val = U + Q
                best_child = child

        self._visit(best_child)


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    board_size = 5
    model = torch.load('models/five_board_wd0.001.pt', map_location=device)#.device) wtf how does this work?

    board = Board(board_size)
    mcts = MCTS(model=model, c_puct=.5, root_board=board, device=device)
    for i in range(0, 1600):
        if i % 100 == 0:
            print(i)
        mcts.search()
    print(torch.Tensor([child.N for child in mcts.root.children]).view(5,5))
    print(torch.Tensor([child.W/child.N for child in mcts.root.children]).view(5, 5))


if __name__ == '__main__':
    test()
