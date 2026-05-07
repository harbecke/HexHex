import random

import torch
from omegaconf import OmegaConf
from hexhex.logic.hexboard import Board
from hexhex.creation.create_model import create_model


def _legacy_board_tensor(logical_board_tensor: torch.Tensor, player: int) -> torch.Tensor:
    """Reference implementation of the bordered + perspective-flipped board tensor.

    Mirrors what `Board.set_border` + the player-1 transpose+roll path used to
    produce. Tests use this as ground truth so any refactor of the in-place
    storage stays bit-equivalent to the legacy formulation.
    """
    n = logical_board_tensor.shape[1]
    bordered = torch.zeros([2, n + 2, n + 2])
    bordered[0, 0, 1:-1] = 1
    bordered[0, -1, 1:-1] = 1
    bordered[1, 1:-1, 0] = 1
    bordered[1, 1:-1, -1] = 1
    bordered[:, 1:-1, 1:-1] = logical_board_tensor
    if player == 1:
        return torch.transpose(torch.roll(bordered, 1, 0), 1, 2)
    return bordered


def test_board_initialization():
    size = 11
    board = Board(size)
    assert board.size == size
    assert board.player == 0
    assert not board.winner
    assert len(board.legal_moves) == size * size

def test_board_move():
    size = 11
    board = Board(size)

    # Make a few moves to bypass special logic for first/second move (switch rule)
    move1 = (5, 5)
    board.set_stone(move1)

    move2 = (0, 0)
    board.set_stone(move2)

    move3 = (1, 1)
    board.set_stone(move3)

    assert board.player == 1
    assert move1 in board.made_moves
    assert move2 in board.made_moves
    assert move3 in board.made_moves

    assert move1 not in board.legal_moves
    assert move2 not in board.legal_moves
    assert move3 not in board.legal_moves


def test_board_tensor_initial_borders():
    board = Board(5)
    bt = board.board_tensor
    assert bt.shape == (2, 7, 7)
    # Player 0 (red) goal edges: top and bottom rows of layer 0 (corners excluded).
    assert torch.all(bt[0, 0, 1:-1] == 1)
    assert torch.all(bt[0, -1, 1:-1] == 1)
    # Player 1 (blue) goal edges: left and right columns of layer 1 (corners excluded).
    assert torch.all(bt[1, 1:-1, 0] == 1)
    assert torch.all(bt[1, 1:-1, -1] == 1)
    # Corners must be zero on both layers.
    for layer in (0, 1):
        for r, c in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
            assert bt[layer, r, c] == 0
    # Cross-layer borders zero (e.g. top row of layer 1 stays 0, left col of layer 0 stays 0).
    assert torch.all(bt[1, 0, :] == 0)
    assert torch.all(bt[1, -1, :] == 0)
    assert torch.all(bt[0, :, 0] == 0)
    assert torch.all(bt[0, :, -1] == 0)
    # Inner 5x5 region empty.
    assert torch.all(bt[:, 1:-1, 1:-1] == 0)


def test_board_tensor_matches_legacy_after_each_move():
    board = Board(7, switch_allowed=False)
    moves = [(3, 3), (1, 1), (4, 4), (0, 6), (5, 2), (2, 0)]
    expected = _legacy_board_tensor(board.logical_board_tensor, board.player)
    assert torch.equal(board.board_tensor, expected)
    for move in moves:
        board.set_stone(move)
        expected = _legacy_board_tensor(board.logical_board_tensor, board.player)
        assert torch.equal(board.board_tensor, expected), f"mismatch after {move}"


def test_board_tensor_random_full_game_matches_legacy():
    rng = random.Random(42)
    board = Board(5)
    while not board.winner and board.legal_moves:
        expected = _legacy_board_tensor(board.logical_board_tensor, board.player)
        assert torch.equal(board.board_tensor, expected)
        move = rng.choice(sorted(board.legal_moves))
        board.set_stone(move)
    expected = _legacy_board_tensor(board.logical_board_tensor, board.player)
    assert torch.equal(board.board_tensor, expected)
    assert board.winner  # the random game eventually finishes


def test_board_tensor_switch_path_matches_legacy():
    board = Board(5, switch_allowed=True)
    board.set_stone((2, 2))  # player 0 plays
    board.set_stone((2, 2))  # player 1 invokes the swap
    assert board.switch
    # Per the switch branch: player is NOT flipped, stays at 1.
    assert board.player == 1
    expected = _legacy_board_tensor(board.logical_board_tensor, board.player)
    assert torch.equal(board.board_tensor, expected)
    # Explicit invariants on the switch encoding.
    bt = board.board_tensor
    # logical[1][2,2] = 0.001 marks the switched cell; layer 0 still has the
    # original stone at logical[0][2,2] = 1. In the player-1 perspective view,
    # these map to swapped coords [1, 3, 3] = 1 and [0, 3, 3] = 0.001.
    assert bt[1, 3, 3] == 1
    assert bt[0, 3, 3] == 0.001


def test_board_tensor_no_switch_first_move_matches_legacy():
    board = Board(5, switch_allowed=False)
    board.set_stone((2, 2))
    expected = _legacy_board_tensor(board.logical_board_tensor, board.player)
    assert torch.equal(board.board_tensor, expected)
    # logical[1][2,2] is set to 0.001 in the switch_allowed=False branch as a
    # marker that the swap option is unavailable; verify it shows in the view.
    bt = board.board_tensor
    assert bt[1, 3, 3] == 1  # original red stone via the perspective-swapped view


def test_undo_move_board_restores_state():
    a = Board(5)
    a.set_stone((2, 2))
    a.set_stone((1, 1))
    a.set_stone((3, 3))

    b = Board(5)
    b.set_stone((2, 2))
    b.set_stone((1, 1))

    a.undo_move_board()
    assert torch.equal(a.board_tensor, b.board_tensor)
    assert torch.equal(a.logical_board_tensor, b.logical_board_tensor)
    assert a.player == b.player
    assert a.made_moves == b.made_moves
    assert a.legal_moves == b.legal_moves


def test_set_stone_immutable_does_not_mutate():
    board = Board(5)
    board.set_stone((2, 2))
    snapshot = board.board_tensor.clone()
    snapshot_logical = board.logical_board_tensor.clone()
    snapshot_player = board.player

    new_board = board.set_stone_immutable((1, 1))

    # Original board untouched.
    assert torch.equal(board.board_tensor, snapshot)
    assert torch.equal(board.logical_board_tensor, snapshot_logical)
    assert board.player == snapshot_player
    # New board has the next move applied.
    assert (1, 1) in new_board.made_moves
    assert new_board.player == 0


def test_model_creation():
    config = OmegaConf.create({
        'board_size': 3,
        'layers': 2,
        'intermediate_channels': 5,
        'reach': 1,
        'switch_model': False,
        'rotation_model': True,
    })

    model = create_model(config)
    assert model.board_size == 3

    # Test forward pass
    input_tensor = torch.zeros((1, 2, 5, 5)) # 3 + 2 for border
    output = model(input_tensor)
    assert output.shape == (1, 9)
