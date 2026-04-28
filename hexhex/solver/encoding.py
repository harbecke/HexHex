"""Position encoding for the Hex solver.

Backend convention (matches hexhex.logic.hexboard): cell index = x * n + y, where
x is the row axis (player 0 / red connects across x, top↔bottom) and y is the
column axis (player 1 / blue connects across y, left↔right). Neighbours of (x, y)
are (x-1, y), (x-1, y+1), (x, y-1), (x, y+1), (x+1, y-1), (x+1, y).

A position is represented internally as two bitmasks (red_mask, blue_mask), each
n*n bits wide. Bit i corresponds to cell index i. The on-disk key is a base-3
integer: digit i is 0 (empty), 1 (red), or 2 (blue).
"""

from functools import lru_cache


def red_to_move(red_mask: int, blue_mask: int) -> bool:
    """Red plays first; equal stone counts means red is to move."""
    return red_mask.bit_count() == blue_mask.bit_count()


def base3_key(red_mask: int, blue_mask: int, n: int) -> int:
    """Encode (red_mask, blue_mask) as a base-3 integer with digit i = cell i."""
    key = 0
    for i in range(n * n - 1, -1, -1):
        key *= 3
        bit = 1 << i
        if red_mask & bit:
            key += 1
        elif blue_mask & bit:
            key += 2
    return key


def decode_key(key: int, n: int) -> tuple[int, int]:
    """Inverse of base3_key."""
    red_mask = 0
    blue_mask = 0
    for i in range(n * n):
        d = key % 3
        if d == 1:
            red_mask |= 1 << i
        elif d == 2:
            blue_mask |= 1 << i
        key //= 3
    return red_mask, blue_mask


@lru_cache(maxsize=None)
def neighbor_masks(n: int) -> tuple[int, ...]:
    """Bitmask of neighbours for each cell index, using backend (x, y) convention."""
    masks = [0] * (n * n)
    deltas = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
    for x in range(n):
        for y in range(n):
            i = x * n + y
            for dx, dy in deltas:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n:
                    masks[i] |= 1 << (nx * n + ny)
    return tuple(masks)


@lru_cache(maxsize=None)
def edge_masks(n: int) -> tuple[int, int, int, int]:
    """Return (red_top, red_bottom, blue_left, blue_right) edge masks.

    Red wins by connecting any red_top cell to any red_bottom cell.
    Blue wins by connecting any blue_left cell to any blue_right cell.
    """
    red_top = 0
    red_bottom = 0
    blue_left = 0
    blue_right = 0
    for y in range(n):
        red_top |= 1 << (0 * n + y)
        red_bottom |= 1 << ((n - 1) * n + y)
    for x in range(n):
        blue_left |= 1 << (x * n + 0)
        blue_right |= 1 << (x * n + (n - 1))
    return red_top, red_bottom, blue_left, blue_right


def player_wins_with_move(player_mask: int, move_idx: int, neighbors: tuple[int, ...],
                          edge_a: int, edge_b: int) -> bool:
    """BFS the player's connected component containing move_idx; check it touches both edges."""
    visited = 0
    frontier = 1 << move_idx
    reached_a = False
    reached_b = False
    while frontier:
        bit = frontier & -frontier
        frontier ^= bit
        if visited & bit:
            continue
        visited |= bit
        if bit & edge_a:
            reached_a = True
        if bit & edge_b:
            reached_b = True
        if reached_a and reached_b:
            return True
        i = bit.bit_length() - 1
        frontier |= neighbors[i] & player_mask & ~visited
    return False


def player_has_won(player_mask: int, neighbors: tuple[int, ...], edge_a: int, edge_b: int) -> bool:
    """True if player_mask contains any connected path between edge_a and edge_b."""
    visited = 0
    frontier = player_mask & edge_a
    while frontier:
        bit = frontier & -frontier
        frontier ^= bit
        if visited & bit:
            continue
        visited |= bit
        if bit & edge_b:
            return True
        i = bit.bit_length() - 1
        frontier |= neighbors[i] & player_mask & ~visited
    return False
