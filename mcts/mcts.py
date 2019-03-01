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
        self._board = None  # will be set upon first call of board(). For root node this needs to be set manually
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = None
        self.children = []
        self.parent = parent
        self.move_idx = move_idx # last move on the board, None for root node

    def move_history(self):
        history = []
        node = self
        while node.parent is not None:
            history.append(node.move_idx)
            node = node.parent
        return history[::-1]

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
        self.P = torch.exp(model_policy)
        for move_idx, p in enumerate(self.P):
            move = to_move(move_idx, self.board().size)
            if move in self.board().legal_moves and p > 1e-6:
                self.children.append(Node(self, move_idx))


class MCTS:
    def __init__(self, model, root_board, args):
        self.model = model.to(device.device)
        self.args = args
        self.root = Node.create_root_node(board=root_board)

    def search(self, batch_size):
        leaf_nodes = []
        for _ in range(batch_size):
            leaf_node = self._visit(self.root)
            if leaf_node:
                leaf_nodes.append(leaf_node)
        self.expand_leafes(leaf_nodes)

    def _backup(self, node: Node, val: float):
        node.N += 1 - self.args.n_virtual_loss
        node.W += val + self.args.n_virtual_loss
        node.Q = node.W / node.N
        if node.parent is not None:
            self._backup(node.parent, -val)


    def _visit(self, node: Node):
        """
        visits node and recursively descends into tree
        If a leaf is encountered, it is returned to be evaluated by the model. backup happens afterwards.
        If a winning move is encountered, backup happens directly and None is returned.
        """

        # prevent node to be visited before backup is run
        node.W += -self.args.n_virtual_loss
        node.N += self.args.n_virtual_loss
        node.Q = node.W / node.N

        if node.has_winner():
            self._backup(node, 1.)  # player has won the game -> value last move with 1
            return None

        if node.is_leaf():
            return node

        best_move = self.find_best_move(node)
        return self._visit(best_move)

    def find_best_move(self, node: Node):
        max_val, best_child = -float("inf"), None

        for child in node.children:
            if node.N == 1:
                # formula from alpha go paper doesn't make much sense for this case.
                # just use model predicted policy here
                U = self.args.c_puct * node.P[child.move_idx]
            else:
                U = self.args.c_puct * node.P[child.move_idx] * np.sqrt(node.N - 1) / (1 + child.N)
                # assert(node.N - 1 == sum(child.N for child in node.children))
            # logger.debug(f'{child.move_idx:2d} {Q:.3f} {U:.3f}')
            if U + child.Q > max_val:
                max_val = U + child.Q
                best_child = child

        return best_child

    def expand_leafes(self, leaf_nodes):
        if len(leaf_nodes) == 0:
            return
        with torch.no_grad():
            board_tensors = torch.stack(
                    [leaf_node.board().board_tensor for leaf_node in leaf_nodes]
            ).to(device.device)
            model_policies, model_values = self.model(board_tensors)
            for idx, node in enumerate(leaf_nodes):
                # leaf_nodes might contain duplicates. don't expand a node twice
                # but we backup v twice
                if node.is_leaf():
                    node.expand(model_policies[idx])
                v = model_values[idx].item()
                self._backup(node, v)


class MCTSSearch:
    def __init__(self, model, args):
        self.model = model
        self.args = args

    def move_counts(self, board):
        """
        Runs a Monte Carlo Tree Search and
        returns visit counts for the next moves.
        """
        mcts = MCTS(model=self.model, root_board=board, args=self.args)
        # initial search with batch_size=1 to allow expanding first node
        mcts.search(batch_size=1)
        for i in range(self.args.num_mcts_simulations // self.args.mcts_batch_size):
            mcts.search(batch_size=self.args.mcts_batch_size)

        counts = [0] * board.size * board.size
        Qs = [0] * board.size * board.size
        for child in mcts.root.children:
            counts[child.move_idx] = child.N
            Qs[child.move_idx] = child.W / child.N if child.N > 0 else 0

        return counts

    @staticmethod
    def move_probabilities(counts, temperature):
        if temperature < 1e-8:
            best_move = np.argmax(counts) # TODO sample over max values
            probs = [0] * len(counts)
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
        'c_puct': 5.,
        'num_mcts_simulations': 1600,
        'mcts_batch_size': 1,
        'n_virtual_loss': 3
    })

    mcts_search = MCTSSearch(model, args)

    board = Board(board_size)
    counts = mcts_search.move_counts(board)
    logger.debug(torch.Tensor(counts).view(5, 5))
    # TODO add noise

if __name__ == '__main__':
    import cProfile
    cProfile.run('test()', 'mcts.profile')
    # test()
