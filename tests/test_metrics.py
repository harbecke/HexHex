import tempfile
from pathlib import Path

from hexhex.logic.hexboard import Board
from hexhex.solver.metrics import OptimalityChecker, board_to_masks
from hexhex.solver.solve import solve, write_table
from hexhex.solver.table import SolutionTable


def _build_table(n: int, tmpdir: Path) -> SolutionTable:
    tt = solve(n)
    path = tmpdir / f"{n}x{n}.bin"
    write_table(tt, n, path)
    return SolutionTable(path)


def test_board_to_masks_empty():
    board = Board(size=3, switch_allowed=False)
    assert board_to_masks(board) == (0, 0)


def test_board_to_masks_after_moves():
    n = 3
    board = Board(size=n, switch_allowed=False)
    board.set_stone((1, 1))   # red at idx 4
    board.set_stone((0, 2))   # blue at idx 2
    red, blue = board_to_masks(board)
    assert red == 1 << 4
    assert blue == 1 << 2


def test_optimality_checker_skips_opening_move():
    """The first move on an empty board is excluded from the metric: it has
    no prior moves whose labels it could influence."""
    n = 3
    with tempfile.TemporaryDirectory() as tmp:
        checker = OptimalityChecker(_build_table(n, Path(tmp)))
    board = Board(size=n, switch_allowed=False)
    for idx in range(n * n):
        assert checker.is_optimal(board, idx) is None


def test_optimality_checker_immediate_win():
    n = 3
    with tempfile.TemporaryDirectory() as tmp:
        checker = OptimalityChecker(_build_table(n, Path(tmp)))
    board = Board(size=n, switch_allowed=False)
    # Sequence: red(0,0), blue(0,2), red(1,0), blue(1,2). Now red plays (2,0) -> connects column 0.
    board.set_stone((0, 0))
    board.set_stone((0, 2))
    board.set_stone((1, 0))
    board.set_stone((1, 2))
    assert board.player == 0  # red to move
    move_idx = 2 * n + 0  # (2, 0) -> idx 6
    assert checker.is_optimal(board, move_idx) is True


def test_optimality_checker_blue_to_move_losing_position_skipped():
    """After red plays the winning centre move, blue is in a losing position;
    the checker should skip every blue response by returning None."""
    n = 3
    with tempfile.TemporaryDirectory() as tmp:
        checker = OptimalityChecker(_build_table(n, Path(tmp)))
    board = Board(size=n, switch_allowed=False)
    board.set_stone((1, 1))  # red plays centre. Now blue to move from a losing position.
    assert board.player == 1
    for idx in range(n * n):
        if idx == 4:
            continue  # occupied
        assert checker.is_optimal(board, idx) is None


def test_optimality_checker_blue_to_move_winning_position():
    """If blue is to move in a position where blue is winning, the checker
    should distinguish optimal from mistake responses."""
    n = 3
    with tempfile.TemporaryDirectory() as tmp:
        checker = OptimalityChecker(_build_table(n, Path(tmp)))
    board = Board(size=n, switch_allowed=False)
    # Red opens with the acute corner (0, 0) — a known losing first move.
    # So blue is now to move in a winning position.
    board.set_stone((0, 0))
    assert board.player == 1
    optimal = sum(1 for idx in range(n * n)
                  if idx != 0 and checker.is_optimal(board, idx) is True)
    mistakes = sum(1 for idx in range(n * n)
                   if idx != 0 and checker.is_optimal(board, idx) is False)
    # Some responses keep blue winning, others throw it away.
    assert optimal > 0
    assert mistakes > 0
    assert optimal + mistakes == n * n - 1


def test_optimality_checker_wrong_size_returns_none():
    n = 3
    with tempfile.TemporaryDirectory() as tmp:
        checker = OptimalityChecker(_build_table(n, Path(tmp)))
    board = Board(size=4, switch_allowed=False)
    assert checker.is_optimal(board, 0) is None


def test_second_move_optimality_count_after_centre_opening():
    """After red opens with the winning centre move, blue is in a losing
    position so every blue response should be skipped (None)."""
    n = 3
    with tempfile.TemporaryDirectory() as tmp:
        checker = OptimalityChecker(_build_table(n, Path(tmp)))
    board = Board(size=n, switch_allowed=False)
    board.set_stone((1, 1))  # red centre
    skipped = sum(1 for idx in range(n * n)
                  if idx != 4 and checker.is_optimal(board, idx) is None)
    assert skipped == n * n - 1
