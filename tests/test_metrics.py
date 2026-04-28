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


def test_optimality_checker_centre_first_move_is_winning():
    n = 3
    with tempfile.TemporaryDirectory() as tmp:
        checker = OptimalityChecker(_build_table(n, Path(tmp)))
    board = Board(size=n, switch_allowed=False)
    # Red plays centre (1, 1) -> idx 4. Known winning first move.
    assert checker.is_optimal(board, 4) is True


def test_optimality_checker_acute_corner_is_losing():
    n = 3
    with tempfile.TemporaryDirectory() as tmp:
        checker = OptimalityChecker(_build_table(n, Path(tmp)))
    board = Board(size=n, switch_allowed=False)
    # (0, 0) is the acute corner, a losing first move.
    assert checker.is_optimal(board, 0) is False


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


def test_optimality_checker_blue_to_move():
    """Verify the checker correctly handles blue-to-move positions without
    needing to invert any perspective transform."""
    n = 3
    with tempfile.TemporaryDirectory() as tmp:
        checker = OptimalityChecker(_build_table(n, Path(tmp)))
    board = Board(size=n, switch_allowed=False)
    board.set_stone((1, 1))  # red plays centre. Now blue to move.
    assert board.player == 1
    # For each possible blue first response, the checker should return True/False
    # consistent with the table — and at least one move must be optimal in any
    # non-terminal position.
    any_optimal = False
    for idx in range(n * n):
        if idx == 4:
            continue  # occupied
        verdict = checker.is_optimal(board, idx)
        assert verdict is not None
        if verdict:
            any_optimal = True
    # Since red's centre move puts the position into a state where red wins,
    # blue is in a losing position — so every blue move is suboptimal.
    # (Centre as red's first move was confirmed to be winning by test_3x3_starting_moves.)
    assert any_optimal is False


def test_optimality_checker_wrong_size_returns_none():
    n = 3
    with tempfile.TemporaryDirectory() as tmp:
        checker = OptimalityChecker(_build_table(n, Path(tmp)))
    board = Board(size=4, switch_allowed=False)
    assert checker.is_optimal(board, 0) is None


def test_first_move_optimality_count_matches_known_pattern():
    """For each of the 9 first moves on 3x3, the checker should agree with the
    pattern from test_3x3_starting_moves: exactly 5 winning moves."""
    n = 3
    with tempfile.TemporaryDirectory() as tmp:
        checker = OptimalityChecker(_build_table(n, Path(tmp)))
    optimal = 0
    for idx in range(n * n):
        board = Board(size=n, switch_allowed=False)
        if checker.is_optimal(board, idx):
            optimal += 1
    assert optimal == 5
