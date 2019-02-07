import torch


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
    the second set is the indices of these stones in the direction of the player
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


class Board():
    '''
    Board is in quadratic shape. This means diagonal neighbours are upper right and lower left, but not the other two.
    First player has to connect top to bottom, second player left to right.
    If the second player decides to switch, a stone is set in the second layer that is only information and not connected. The second player now plays the first layer and vice-versa.
    '''
    def __init__(self, size):
        self.size = size
        self.board_tensor = torch.zeros([2, self.size, self.size])
        self.legal_moves = set([(idx1, idx2) for idx1 in range(self.size) for idx2 in range(self.size)])
        self.illegal_moves = set()
        self.connected_sets = [[], []]
        self.switch = False
        self.winner = False

    def __repr__(self):
        return ('Board\n'+str((self.board_tensor[0]-self.board_tensor[1]).numpy())
        +'\nLegal moves\n'+str(self.legal_moves)
        +'\nIllegal moves\n'+str(self.illegal_moves)
        +'\nWinner\n'+str(self.winner)
        +'\nConnected sets\n'+str(self.connected_sets))

    def set_stone(self, player, position):
        if position not in self.illegal_moves:
            self.legal_moves.remove(position)
            self.illegal_moves.update([position])
            self.board_tensor[player][position] = 1
            self.connected_sets[player], self.winner = update_connected_sets_check_win(self.connected_sets[player], player, position, self.size)
            self.switch = False
        elif self.switch==False and set([position])==self.illegal_moves:
            self.switch = True
            self.board_tensor[1][position] = 1
        else:
            print('Illegal Move!')
            print(self)