"""Ground-truth optimality lookup for a self-play move.

Given a `Board` and a move index in original coordinates, decides whether the
move leads to a position where the opponent loses (per a solved-position table).
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
        """Return True iff playing `move_idx` (in original coords) at `board` leads to a
        position where the opponent loses. None if the resulting position isn't in the
        table (shouldn't happen for reachable positions on the right size board)."""
        if board.size != self.table.board_size:
            return None
        red_mask, blue_mask = board_to_masks(board)
        red_to_move = (board.player == 0)
        if red_to_move:
            new_red = red_mask | (1 << move_idx)
            new_blue = blue_mask
            if player_wins_with_move(new_red, move_idx, self.neighbors,
                                     self.red_top, self.red_bottom):
                return True
        else:
            new_red = red_mask
            new_blue = blue_mask | (1 << move_idx)
            if player_wins_with_move(new_blue, move_idx, self.neighbors,
                                     self.blue_left, self.blue_right):
                return True

        key = base3_key(new_red, new_blue, self.table.board_size)
        if key not in self.table:
            return None
        return not self.table.winning(key)
