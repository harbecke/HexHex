import tempfile
from pathlib import Path

import pytest

from hexhex.solver.encoding import (
    base3_key,
    decode_key,
    edge_masks,
    neighbor_masks,
    player_has_won,
    player_wins_with_move,
    red_to_move,
)
from hexhex.solver.solve import solve, write_table
from hexhex.solver.table import SolutionTable


def test_base3_key_roundtrip():
    n = 3
    # Red on (0,0), blue on (1,1), red on (2,2).
    red = (1 << 0) | (1 << 8)
    blue = 1 << 4
    key = base3_key(red, blue, n)
    assert decode_key(key, n) == (red, blue)


def test_red_to_move():
    assert red_to_move(0, 0)
    assert not red_to_move(0b1, 0)
    assert red_to_move(0b1, 0b10)


def test_neighbor_masks_3x3_centre():
    # Centre cell (1, 1) -> idx 4. Neighbours by delta (-1,0)(-1,1)(0,-1)(0,1)(1,-1)(1,0)
    # are cells idx 1, 2, 3, 5, 6, 7.
    nbrs = neighbor_masks(3)
    expected = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 5) | (1 << 6) | (1 << 7)
    assert nbrs[4] == expected


def test_player_win_detection_3x3_red_column():
    # Red plays (0,0)=idx0 -> (1,0)=idx3 -> (2,0)=idx6: column 0, top to bottom.
    n = 3
    nbrs = neighbor_masks(n)
    red_top, red_bottom, _, _ = edge_masks(n)
    red = (1 << 0) | (1 << 3) | (1 << 6)
    assert player_wins_with_move(red, 6, nbrs, red_top, red_bottom)


def test_player_win_detection_3x3_no_win():
    n = 3
    nbrs = neighbor_masks(n)
    red_top, red_bottom, _, _ = edge_masks(n)
    red = (1 << 0) | (1 << 3)  # only top two rows of column 0
    assert not player_wins_with_move(red, 3, nbrs, red_top, red_bottom)


def test_3x3_red_wins_from_empty():
    """3x3 Hex with no pie rule: first player (red) wins."""
    tt = solve(3)
    assert tt[0] is True  # empty board, red to move


def test_3x3_table_roundtrip():
    tt = solve(3)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "3x3.bin"
        write_table(tt, 3, path)
        table = SolutionTable(path)
    assert table.board_size == 3
    assert table.num_entries == len(tt)
    # Empty board should be in the table and red should be winning.
    assert table.winning_from_masks(0, 0) is True


@pytest.mark.parametrize("n", [3])
def test_solver_position_invariants(n):
    """Every stored position has stone counts consistent with red-first play."""
    tt = solve(n)
    shift = n * n
    for packed in tt:
        red_mask = packed >> shift
        blue_mask = packed & ((1 << shift) - 1)
        rc = red_mask.bit_count()
        bc = blue_mask.bit_count()
        # Red plays first, so |red| == |blue| (red to move) or |red| == |blue| + 1.
        assert rc == bc or rc == bc + 1
        assert (red_mask & blue_mask) == 0  # no overlap


def test_3x3_starting_moves():
    """Map each of red's first moves to win/loss for red, and check symmetry.

    A first move at cell i is a red-win iff blue (now to move) loses from the
    resulting position — i.e. tt[child] is False. The 180° rotation
    (x, y) -> (n-1-x, n-1-y) swaps each player's two edges with themselves, so
    a position and its rotation share the same winner. Therefore each first
    move's outcome equals that of its rotated counterpart.
    """
    n = 3
    tt = solve(n)
    shift = n * n

    outcomes: dict[tuple[int, int], bool] = {}
    for x in range(n):
        for y in range(n):
            i = x * n + y
            child_packed = ((1 << i) << shift) | 0
            assert child_packed in tt, f"first move ({x},{y}) child not in TT"
            blue_wins_from_child = tt[child_packed]
            red_wins_after_move = not blue_wins_from_child
            outcomes[(x, y)] = red_wins_after_move

    # 180° rotation symmetry.
    for (x, y), win in outcomes.items():
        assert outcomes[(n - 1 - x, n - 1 - y)] == win, f"asymmetry at ({x},{y})"

    # Strategy-stealing: at least one first move must win for red, and the
    # empty-board solver result must agree.
    assert any(outcomes.values())
    assert tt[0] == any(outcomes.values())

    # Concrete pattern on 3x3 with no pie rule. The hex shear makes the (0, n-1)
    # and (n-1, 0) corners "obtuse" (3 neighbours) while (0, 0) and (n-1, n-1)
    # are "acute" (only 2 neighbours), so the obtuse corners are stronger.
    # Red wins from: centre, the obtuse corners, and the middle row {(1, *)}.
    # Red loses from: the acute corners and the top/bottom edge midpoints {(0,1),(2,1)}.
    expected = {
        (0, 0): False, (0, 1): False, (0, 2): True,
        (1, 0): True,  (1, 1): True,  (1, 2): True,
        (2, 0): True,  (2, 1): False, (2, 2): False,
    }
    assert outcomes == expected


def test_3x3_full_coverage_via_negamax():
    """Every non-terminal stored position satisfies the negamax invariant:
    side-to-move wins iff some legal move yields immediate win OR a child where
    opponent loses. Every reachable child of a stored non-terminal position must
    itself be in the table."""
    n = 3
    tt = solve(n)
    nbrs = neighbor_masks(n)
    red_top, red_bottom, blue_left, blue_right = edge_masks(n)
    shift = n * n
    cells_mask = (1 << shift) - 1

    for packed, winning in tt.items():
        red_mask = packed >> shift
        blue_mask = packed & cells_mask
        # Skip terminal positions (someone has already won) — they have no legal
        # successors and the solver stores them as `False` placeholders.
        if player_has_won(red_mask, nbrs, red_top, red_bottom):
            assert winning is False
            continue
        if player_has_won(blue_mask, nbrs, blue_left, blue_right):
            assert winning is False
            continue
        red_turn = red_to_move(red_mask, blue_mask)
        free = cells_mask & ~(red_mask | blue_mask)
        if not free:
            continue
        derived_winning = False
        while free:
            bit = free & -free
            free ^= bit
            i = bit.bit_length() - 1
            if red_turn:
                new_red = red_mask | bit
                child_packed = (new_red << shift) | blue_mask
                if player_wins_with_move(new_red, i, nbrs, red_top, red_bottom):
                    derived_winning = True
                else:
                    assert child_packed in tt, f"unreachable child {child_packed} from {packed}"
                    if not tt[child_packed]:
                        derived_winning = True
            else:
                new_blue = blue_mask | bit
                child_packed = (red_mask << shift) | new_blue
                if player_wins_with_move(new_blue, i, nbrs, blue_left, blue_right):
                    derived_winning = True
                else:
                    assert child_packed in tt, f"unreachable child {child_packed} from {packed}"
                    if not tt[child_packed]:
                        derived_winning = True
        # If this position is a non-terminal (someone hasn't won yet), the stored
        # value must equal the derived value.
        assert winning == derived_winning, f"mismatch at {packed}: stored={winning} derived={derived_winning}"
