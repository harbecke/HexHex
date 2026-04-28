"""Ground-truth optimality lookup for a self-play move.

Given a `Board` and a move index in original coordinates, decides whether the
move was a mistake. Only moves from winning positions are judged: if the
player-to-move is already in a losing position then every move loses, so move
quality is undefined and we skip the position.
"""

from __future__ import annotations

from hexhex.logic.hexboard import Board
from hexhex.solver.encoding import (
    base3_key,
    edge_masks,
    neighbor_masks,
    player_wins_with_move,
)
from hexhex.solver.table import SolutionTable


def board_to_masks(board: Board) -> tuple[int, int]:
    """Read (red_mask, blue_mask) directly from `Board.logical_board_tensor`.

    `logical_board_tensor[player, x, y]` is the canonical 2-channel stone layout
    that is *not* affected by the perspective swap applied to `board_tensor`.
    """
    layer = board.logical_board_tensor
    n = board.size
    red_mask = 0
    blue_mask = 0
    for x in range(n):
        for y in range(n):
            i = x * n + y
            if layer[0, x, y].item() > 0.5:
                red_mask |= 1 << i
            if layer[1, x, y].item() > 0.5:
                blue_mask |= 1 << i
    return red_mask, blue_mask


class OptimalityChecker:
    """Caches per-board-size lookup state so it can be reused across many calls."""

    def __init__(self, table: SolutionTable):
        self.table = table
        n = table.board_size
        self.neighbors = neighbor_masks(n)
        self.red_top, self.red_bottom, self.blue_left, self.blue_right = edge_masks(n)

    def is_optimal(self, board: Board, move_idx: int) -> bool | None:
        """Judge `move_idx` (in original coords) at `board` against the solved table.

        Returns:
            True  — winning move from a winning position (optimal).
            False — losing move from a winning position (mistake).
            None  — the position is a loss for the side-to-move (skip; every move
                    loses so move quality is undefined), or the position/board size
                    isn't covered by the table.

        Raises AssertionError if a move appears to win from a losing position —
        that would mean the table is inconsistent.
        """
        if board.size != self.table.board_size:
            return None
        n = self.table.board_size
        red_mask, blue_mask = board_to_masks(board)
        red_to_move = (board.player == 0)

        # Skip the opening move on an empty board: there are no prior moves whose
        # labels it could influence, so its quality has no effect on training signal.
        if red_mask == 0 and blue_mask == 0:
            return None

        cur_key = base3_key(red_mask, blue_mask, n)
        if cur_key not in self.table:
            return None
        cur_winning = self.table.winning(cur_key)

        if red_to_move:
            new_red = red_mask | (1 << move_idx)
            new_blue = blue_mask
            immediate_win = player_wins_with_move(
                new_red, move_idx, self.neighbors, self.red_top, self.red_bottom)
        else:
            new_red = red_mask
            new_blue = blue_mask | (1 << move_idx)
            immediate_win = player_wins_with_move(
                new_blue, move_idx, self.neighbors, self.blue_left, self.blue_right)

        if immediate_win:
            assert cur_winning, "winning move from a losing position — solver table bug"
            return True

        if not cur_winning:
            return None

        new_key = base3_key(new_red, new_blue, n)
        if new_key not in self.table:
            return None
        # opponent is now to move; if opponent loses the move kept us winning
        return not self.table.winning(new_key)
