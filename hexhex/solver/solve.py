"""Negamax solver for Hex (no pie rule).

Solves every reachable position from a given root and writes a sorted-keys
binary table consumable by hexhex.solver.table.SolutionTable.

Stores results as `(red_mask, blue_mask) -> bool` (current player wins) in an
in-memory transposition table. At the end the TT is converted to base-3 keys,
sorted, and dumped.

CLI:
    uv run python -m hexhex.solver.solve --size 3 --out tables/3x3.bin
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

from hexhex.solver.encoding import (
    base3_key,
    edge_masks,
    neighbor_masks,
    player_wins_with_move,
    red_to_move,
)

MAGIC = b"HXSV"
HEADER_FMT = "<4sHQ"  # magic, board_size (u16), num_entries (u64)


def _move_order(n: int) -> list[int]:
    """Cells ordered by distance from the centre (centre first). Helps the solver
    find a winning move sooner and prune deeper."""
    cx = (n - 1) / 2
    cells = []
    for x in range(n):
        for y in range(n):
            d = (x - cx) ** 2 + (y - cx) ** 2
            cells.append((d, x * n + y))
    cells.sort()
    return [idx for _, idx in cells]


def solve(n: int) -> dict[int, bool]:
    """Solve the game from the empty board. Returns a TT mapping packed
    (red_mask << n^2 | blue_mask) -> True if side-to-move wins."""
    neighbors = neighbor_masks(n)
    red_top, red_bottom, blue_left, blue_right = edge_masks(n)
    cells = (1 << (n * n)) - 1
    order = _move_order(n)
    shift = n * n
    tt: dict[int, bool] = {}

    sys.setrecursionlimit(max(10_000, n * n * 4))

    def recurse(red_mask: int, blue_mask: int) -> bool:
        packed = (red_mask << shift) | blue_mask
        cached = tt.get(packed)
        if cached is not None:
            return cached
        red_turn = red_to_move(red_mask, blue_mask)
        occupied = red_mask | blue_mask
        has_winning_move = False
        for i in order:
            bit = 1 << i
            if occupied & bit:
                continue
            if red_turn:
                new_red = red_mask | bit
                child_packed = (new_red << shift) | blue_mask
                if player_wins_with_move(new_red, i, neighbors, red_top, red_bottom):
                    # Terminal child: blue is "to move" but red has already won, so blue loses.
                    if child_packed not in tt:
                        tt[child_packed] = False
                    has_winning_move = True
                else:
                    if not recurse(new_red, blue_mask):
                        has_winning_move = True
            else:
                new_blue = blue_mask | bit
                child_packed = (red_mask << shift) | new_blue
                if player_wins_with_move(new_blue, i, neighbors, blue_left, blue_right):
                    if child_packed not in tt:
                        tt[child_packed] = False
                    has_winning_move = True
                else:
                    if not recurse(red_mask, new_blue):
                        has_winning_move = True
        tt[packed] = has_winning_move
        return has_winning_move

    recurse(0, 0)
    return tt


def write_table(tt: dict[int, bool], n: int, path: Path) -> None:
    """Convert a packed-key TT to base-3 keys and dump as sorted u64[] + bitarray."""
    shift = n * n
    blue_mask_bits = (1 << shift) - 1

    keys = np.empty(len(tt), dtype=np.uint64)
    values = np.empty(len(tt), dtype=bool)
    for i, (packed, win) in enumerate(tt.items()):
        red_mask = packed >> shift
        blue_mask = packed & blue_mask_bits
        keys[i] = base3_key(red_mask, blue_mask, n)
        values[i] = win

    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    values = values[order]
    bits = np.packbits(values, bitorder="little")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack(HEADER_FMT, MAGIC, n, len(tt)))
        f.write(keys.tobytes())
        f.write(bits.tobytes())


def _human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def _starting_moves_map(tt: dict[int, bool], n: int) -> str:
    """Render a 2D map of red's first-move outcomes (W = red wins, L = red loses).
    Rows are sheared to evoke the hex geometry: each row indents one half-step
    further right than the row above."""
    shift = n * n
    rows = []
    header = "    " + "  ".join(f"y={y}" for y in range(n))
    rows.append(header)
    for x in range(n):
        cells = []
        for y in range(n):
            i = x * n + y
            child_packed = ((1 << i) << shift) | 0
            blue_wins = tt[child_packed]
            cells.append(" W " if not blue_wins else " L ")
        indent = " " * x
        rows.append(f"{indent}x={x} " + " ".join(cells))
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True, help="Board side length")
    parser.add_argument("--out", type=Path, required=True, help="Output table path")
    args = parser.parse_args()

    t0 = time.time()
    tt = solve(args.size)
    t_solve = time.time() - t0
    print(f"solved {args.size}x{args.size}: {len(tt):,} positions in {t_solve:.2f}s")

    t0 = time.time()
    write_table(tt, args.size, args.out)
    t_write = time.time() - t0
    size_bytes = args.out.stat().st_size
    print(f"wrote {args.out}: {len(tt):,} entries, "
          f"{_human_bytes(size_bytes)} ({size_bytes:,} bytes) in {t_write:.2f}s")

    root_win = tt[0]
    print(f"empty-board result: side-to-move (red) {'wins' if root_win else 'loses'}")
    print()
    print("Starting moves (W = red wins, L = red loses):")
    print(_starting_moves_map(tt, args.size))


if __name__ == "__main__":
    main()
