"""Lookup interface for solved-position tables produced by hexhex.solver.solve."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from hexhex.solver.encoding import base3_key

MAGIC = b"HXSV"
HEADER_FMT = "<4sHQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


class SolutionTable:
    """Maps every reachable position to a win/loss bit for the side-to-move.

    File layout:
        - header: 4-byte magic ("HXSV"), u16 board_size, u64 num_entries
        - sorted u64 keys (base-3 encoded positions)
        - packed bitarray of values (bit i = result for keys[i]), little-bit-order
    """

    def __init__(self, path: str | Path):
        path = Path(path)
        with open(path, "rb") as f:
            header = f.read(HEADER_SIZE)
            magic, board_size, num_entries = struct.unpack(HEADER_FMT, header)
            if magic != MAGIC:
                raise ValueError(f"bad magic in {path}: {magic!r}")
            self.board_size = int(board_size)
            self.num_entries = int(num_entries)
            self.keys = np.frombuffer(f.read(num_entries * 8), dtype=np.uint64)
            num_value_bytes = (num_entries + 7) // 8
            self.values = np.frombuffer(f.read(num_value_bytes), dtype=np.uint8)

    def _index(self, key: int) -> int | None:
        idx = int(np.searchsorted(self.keys, np.uint64(key)))
        if idx >= self.num_entries or int(self.keys[idx]) != key:
            return None
        return idx

    def __contains__(self, key: int) -> bool:
        return self._index(key) is not None

    def winning(self, key: int) -> bool:
        """Return True if the side-to-move wins at this position. KeyError if absent."""
        idx = self._index(key)
        if idx is None:
            raise KeyError(key)
        return bool(self.values[idx >> 3] & (1 << (idx & 7)))

    def winning_from_masks(self, red_mask: int, blue_mask: int) -> bool:
        return self.winning(base3_key(red_mask, blue_mask, self.board_size))
