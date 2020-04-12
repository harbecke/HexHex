import copy

import torch

from hexhex.utils.logger import logger


def to_move_idx(move, board_size):
    return move[0] * board_size + move[1]

def to_move(move_idx, board_size):
    return move_idx // board_size, move_idx % board_size

def position_to_alpha_numeric(position):
    x, y = position
    return chr(97 + y), x + 1

def get_neighbours(position, size):
    x = int(position[0])
    y = int(position[1])

    assert 0 <= x < size
    assert 0 <= y < size

    neighbours = set([(x-1, y), (x-1, y+1), (x, y-1), (x, y+1), (x+1, y-1), (x+1, y)])

    if x == 0:
        neighbours.discard((x-1, y))
        neighbours.discard((x-1, y+1))
    elif x == size-1:
        neighbours.discard((x+1, y-1))
        neighbours.discard((x+1, y))

    if y == 0:
        neighbours.discard((x, y-1))
        neighbours.discard((x+1, y-1))
    elif y == size-1:
        neighbours.discard((x-1, y+1))
        neighbours.discard((x, y+1))

    return neighbours


def update_connected_sets_check_win(connected_sets, player, position, size):
    '''
    save a set of tuple of sets:
    the first set is a connected component of stones
    the second set is the indices of these stones in the direction of the player > change to intervall [,]
    thus the winning condition is having 0 and (size-1) in one of the second sets
    '''
    new_connected_sets = []

    if player == 0:
        new_connected_set = (set([position]), set([position[0]]))
    elif player == 1:
        new_connected_set = (set([position]), set([position[1]]))

    neighbours = get_neighbours(position, size)

    for connected_set in connected_sets:
        if not connected_set[0].isdisjoint(neighbours):
            new_connected_set[0].update(connected_set[0])
            new_connected_set[1].update(connected_set[1])
        else:
            new_connected_sets.append(connected_set)

    new_connected_sets.append(new_connected_set)

    if set([0, size-1]).issubset(new_connected_set[1]):
        return new_connected_sets, [player]
    else:
        return new_connected_sets, False


class Board:
    """
    Board is in quadratic shape. This means diagonal neighbours are upper right and lower left, but not the other two.
    There are three layers: The first layer is for stones of the first player, second layer for second player and the third layer stores indicates whose turn it is.
    First player has to connect his stones on the first dimension (displayed top to bottom), second player on the second dimension (displayed left to right).
    If the second player decides to switch, a stone is set in the second layer that is only information.
    The second player becomes the first player and now plays the first layer and vice-versa.
    """
    def __init__(self, size, switch_allowed=True):
        self.size = size
        self.board_tensor = torch.zeros([2, self.size, self.size])
        self.logical_board_tensor = torch.zeros([2, self.size, self.size])
        self.made_moves = set()
        self.legal_moves = set([(idx1, idx2) for idx1 in range(self.size) for idx2 in range(self.size)])
        self.connected_sets = [[], []]
        self.player = 0
        self.switch = False
        self.winner = False
        self.switch_allowed = switch_allowed
        self.move_history = []

    def override(self, other):
        self.size = other.size
        self.board_tensor = other.board_tensor
        self.logical_board_tensor = other.logical_board_tensor
        self.made_moves = other.made_moves
        self.legal_moves = other.legal_moves
        self.connected_sets = other.connected_sets
        self.player = other.player
        self.switch = other.switch
        self.winner = other.winner
        self.move_history = other.move_history

    def __repr__(self):
        return ('Board\n'+str((self.board_tensor[0]-self.board_tensor[1]).numpy())
            +'\nMade moves\n'+str(self.made_moves)
            +'\nLegal moves\n'+str(self.legal_moves)
            +'\nWinner\n'+str(self.winner)
            +'\nConnected sets\n'+str(self.connected_sets))+'\n'

    def set_stone_immutable(self, position):
        """
        Same as set_stone but does not alter the board.
        Instead it returns a modified deep copy of the board with the stone set.
        """
        self_copy = copy.deepcopy(self)
        self_copy.set_stone(position)
        return self_copy

    def set_stone(self, position):
        if type(position) is int:
            position = to_move(position, self.size)

        if position in self.legal_moves:

            if len(self.made_moves) > 1:
                self.legal_moves.remove(position)

            elif len(self.made_moves) == 1:
                if set([position]) == self.made_moves:
                    self.switch = True
                    self.legal_moves.remove(position)
                    self.logical_board_tensor[1][position] = 0.001
                    self.board_tensor = torch.transpose(torch.roll(self.logical_board_tensor, 1, 0), 1, 2)
                    self.move_history.append((self.player, position))
                    return

                else:
                    self.legal_moves -= self.made_moves
                    self.legal_moves.remove(position)

            elif len(self.made_moves) == 0 and not self.switch_allowed:
                self.legal_moves.remove(position)
                self.logical_board_tensor[1][position] = 0.001

            self.made_moves.update([position])
            self.logical_board_tensor[self.player][position] = 1
            self.connected_sets[self.player], self.winner = update_connected_sets_check_win( \
                self.connected_sets[self.player], self.player, position, self.size)
            self.move_history.append((self.player, position))

            if self.winner:
                if self.switch:
                    self.winner = [[1], [0]][self.winner[0]]
                self.legal_moves = set()

            self.player = 1-self.player
            if self.player:
                self.board_tensor = torch.transpose(torch.roll(self.logical_board_tensor, 1, 0), 1, 2)
            else:
                self.board_tensor = self.logical_board_tensor

        else:
            logger.error(f'Illegal Move! {position} of type {type(position)}')
            logger.error(self)
            logger.error(self.move_history)
            raise SystemExit

    def get_owner(self, position):
        if position[0] in [-1, self.size]:
            return 0
        if position[1] in [-1, self.size]:
            return 1
        if self.logical_board_tensor[0][position].item() == 1:
            return 0
        elif self.logical_board_tensor[1][position].item() == 1:
            return 1
        else:
            return None

    def export_as_FF4(self, filename):
        with open(filename, 'w') as file:
            print(f"(;AP[HexGui:0.9.0]FF[4]GM[11]SZ[{self.size}]", file=file)

            for player, position in self.move_history:
                stone_color = 'B' if player == 0 else 'W'
                alpha, numeric = position_to_alpha_numeric(position)
                print(f';{stone_color}[{alpha}{numeric}]', file=file)

            print(")", file=file)

    def undo_move_board(self):
        new_board = Board(self.size)
        for move in self.move_history[:-1]:
            new_board.set_stone(move[1])
        self.override(new_board)


def all_moves(board_size):
    return [(x, y) for x in range(board_size) for y in range(board_size)]

def first_k_moves(board_size, num_moves):
    if num_moves == 1:
        for move in all_moves(board_size):
            yield [move]
    else:
        for first_moves in first_k_moves(board_size, num_moves - 1):
            for next_move in all_moves(board_size):
                if next_move not in first_moves or len(first_moves)==1:
                    yield first_moves + [next_move]

def get_opened_board(board_size, opening):
    board = Board(size=board_size)
    for position in opening:
        board.set_stone(position)
    return board
