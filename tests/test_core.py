import torch
from omegaconf import OmegaConf
from hexhex.logic.hexboard import Board
from hexhex.creation.create_model import create_model

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
