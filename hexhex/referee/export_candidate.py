"""Wrap a HexHex checkpoint in the referee's candidate contract and export it to ONNX.

For tests and sanity checks, and for sending our own models through the referee. Don't give this file to the
agent: the adapter spells out HexHex's input encoding.

    uv run python -m hexhex.referee.export_candidate reference_models/11_2w4_2000.pt 11_2w4_2000_contract.onnx
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from hexhex.utils.utils import load_cpu_model


class ContractAdapter(nn.Module):
    """Contract input `[B, 2, n, n]` → HexHex board `[B, 2, n+2, n+2]` → model logits `[B, n*n]`.

    `model` should be built with `export_mode=False` so it keeps rotation averaging and its own legality
    mask; the referee masks illegal cells again anyway. Border values match `Board.__init__`.
    """

    def __init__(self, model, n):
        super().__init__()
        self.model = model
        self.board_size = n
        border = torch.zeros(1, 2, n + 2, n + 2)
        border[0, 0, 0, 1:-1] = 1  # side to move: edges of the first spatial axis
        border[0, 0, -1, 1:-1] = 1
        border[0, 1, 1:-1, 0] = 1  # opponent: edges of the second spatial axis
        border[0, 1, 1:-1, -1] = 1
        self.register_buffer("border", border)

    def forward(self, board):
        return self.model(F.pad(board, (1, 1, 1, 1)) + self.border)


def export_onnx(model, path):
    """Export a CPU `model` (see `utils.load_cpu_model`) through the adapter as a single self-contained file."""
    adapter = ContractAdapter(model, model.board_size).eval()
    n = model.board_size
    torch.onnx.export(adapter, (torch.zeros(2, 2, n, n),), path, dynamo=True, external_data=False,
                      input_names=["board"], output_names=["logits"],
                      dynamic_shapes={"board": {0: torch.export.Dim("B")}})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", help="HexHex .pt checkpoint")
    parser.add_argument("out", help="output .onnx path")
    args = parser.parse_args()
    export_onnx(load_cpu_model(args.checkpoint), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
