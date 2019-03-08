#!/usr/bin/env python3
import numpy as np
import torch
import device
from hexboard import Board, to_move
from hexconvolution import MCTSModel
from utils import dotdict
from logger import logger

class Node:
    def __init__(self, parent, move_idx, board_size):
        self._board = None  # will be set upon first call of board(). For root node this needs to be set manually
        self.P = None
        self.N = 0
        self.child_Ns = torch.zeros(board_size*board_size)
        self.child_Ws = torch.zeros(board_size*board_size)
        self.child_Qs = torch.zeros(board_size*board_size)
        self.children = [None for _ in range(board_size * board_size)]
        self.parent = parent
        self.move_idx = move_idx # last move on the board, None for root node
        self._is_leaf = True

    def move_history(self):
        history = []
        node = self
        while node.parent is not None:
            history.append(node.move_idx)
            node = node.parent
        return history[::-1]

    @staticmethod
    def create_root_node(board):
        root = Node(parent=None, move_idx=None, board_size=board.size)
        root._board = board
        return root

    def board(self):
        if not self._board:
            self._board = self.parent.board().set_stone_immutable(to_move(self.move_idx, self.parent.board().size))
        return self._board

    def has_winner(self):
        return self.board().winner != False

    def is_leaf(self):
        return self._is_leaf

    def expand(self, model_policy):
        self._is_leaf = False
        self.P = torch.exp(model_policy)
        for move_idx, p in enumerate(self.P):
            move = to_move(move_idx, self.board().size)
            if move in self.board().legal_moves:
                self.children[move_idx] = Node(self, move_idx, self.board().size)
            else:
                self.P[move_idx] = 0.0
                self.child_Qs[move_idx] = -1e6 # avoid picking this move in search
        self.P /= torch.sum(self.P) # normalize to remove p components from illegal moves



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

    def _backup(self, node: Node, val: float, child = None):
        node.N += 1 - self.args.n_virtual_loss

        if child:
            node.child_Ns[child.move_idx] += 1 - self.args.n_virtual_loss
            node.child_Ws[child.move_idx] += val + self.args.n_virtual_loss
            node.child_Qs[child.move_idx] = node.child_Ws[child.move_idx] / node.child_Ns[child.move_idx]

        if node.parent is not None:
            self._backup(node.parent, -val, node)


    def _visit(self, node: Node):
        """
        visits node and recursively descends into tree
        If a leaf is encountered, it is returned to be evaluated by the model. backup happens afterwards.
        If a winning move is encountered, backup happens directly and None is returned.
        """

        # prevent node to be visited before backup is run
        node.N += self.args.n_virtual_loss
        if node.parent:
            node.parent.child_Ns[node.move_idx] += self.args.n_virtual_loss
            node.parent.child_Ws[node.move_idx] += -self.args.n_virtual_loss
            node.parent.child_Qs[node.move_idx] = node.parent.child_Ws[node.move_idx] / node.parent.child_Ns[node.move_idx]
            assert(node.parent.child_Ns[node.move_idx] == node.N)

        if node.has_winner():
            self._backup(node, 1.)  # player has won the game -> value last move with 1
            return None

        if node.is_leaf():
            return node

        best_move_node = self.find_best_move(node)
        return self._visit(best_move_node)

    def find_best_move(self, node: Node):
        # formula from alpha go paper doesn't make much sense for the case node.N == 1
        # just use model predicted policy there

        if node.N == 1:
            child_ratings = node.child_Qs + self.args.c_puct * node.P
        else:
            N_factor = np.sqrt(node.N - 1)
            child_ratings = node.child_Qs + self.args.c_puct * node.P * N_factor / (1 + node.child_Ns)

        best_move_idx = np.argmax(child_ratings).item()
        assert(node.child[best_move_idx] is not None)
        return node.children[best_move_idx]

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

    def simulate(self, board):
        """
        Runs a Monte Carlo Tree Search and
        returns visit counts for the next moves.
        """
        mcts = MCTS(model=self.model, root_board=board, args=self.args)
        # initial search with batch_size=1 to allow expanding first node
        mcts.search(batch_size=1)
        for i in range(self.args.num_mcts_simulations // self.args.mcts_batch_size):
            mcts.search(batch_size=self.args.mcts_batch_size)

        counts = mcts.root.child_Ns
        Qs = mcts.root.child_Qs

        return counts, Qs

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

    @staticmethod
    def sample_move(move_probabilities):
        move_probabilities = np.array(move_probabilities)

        if np.abs(move_probabilities.sum() - 1) > 1e-4:
            logger.error(f'move probabilities sum up to value != 1: {move_probabilities}, sum = {move_probabilities.sum()}')
            exit(1)

        # numpy is *really* picky about these values summing up to 1 with high precision, so we better make sure they actually do
        move_probabilities /= move_probabilities.sum()
        return np.random.choice(range(len(move_probabilities)), p=move_probabilities)

def test():
    board_size = 5
    model = MCTSModel(board_size=5, layers=10, intermediate_channels=32)

    args = dotdict({
        'c_puct': 5.,
        'num_mcts_simulations': 1600,
        'mcts_batch_size': 8,
        'n_virtual_loss': 3
    })

    mcts_search = MCTSSearch(model, args)

    board = Board(board_size)
    counts, Qs = mcts_search.simulate(board)
    logger.debug(torch.Tensor(counts).view(5, 5))
    # TODO add noise

if __name__ == '__main__':
    import cProfile
    cProfile.run('test()', 'mcts.profile')
    import pstats
    p = pstats.Stats('mcts.profile')
    p.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)
