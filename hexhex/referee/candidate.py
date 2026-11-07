"""Validate uploaded ONNX candidates and wrap them as players that `play_games` accepts.

Contract (see README.md): one float32 input `[B, 2, n, n]` (channel 0 = side to move, oriented so the side to
move connects along dim 2), one float32 output `[B, n*n]` of move scores. Every rejection raises
`CandidateError` with a short, fixed reason that is safe to return to the agent.
"""
import numpy as np
import onnx
import onnxruntime as ort
import torch

# ONNX Runtime prints device-discovery and optimizer warnings to stderr at session creation.
ort.set_default_logger_severity(3)

STANDARD_DOMAINS = ("", "ai.onnx")


class CandidateError(ValueError):
    """A candidate model broke the contract. The message is returned to the agent verbatim."""


def _walk(model):
    """Return every node and every tensor in `model`.

    Covers the main graph, training-info graphs, local functions, and all subgraphs nested in node
    attributes (If/Loop/Scan bodies). Tensors include initializers, sparse initializers and tensor-valued
    attributes such as `Constant` values.
    """
    nodes, tensors = [], []
    graphs = [model.graph]
    for info in model.training_info:
        graphs += [info.initialization, info.algorithm]
    pending = [node for function in model.functions for node in function.node]
    while graphs or pending:
        if graphs:
            graph = graphs.pop()
            tensors.extend(graph.initializer)
            for sparse in graph.sparse_initializer:
                tensors += [sparse.values, sparse.indices]
            pending.extend(graph.node)
        while pending:
            node = pending.pop()
            nodes.append(node)
            for attr in node.attribute:
                if attr.HasField("t"):
                    tensors.append(attr.t)
                tensors.extend(attr.tensors)
                sparse_tensors = list(attr.sparse_tensors)
                if attr.HasField("sparse_tensor"):
                    sparse_tensors.append(attr.sparse_tensor)
                for sparse in sparse_tensors:
                    tensors += [sparse.values, sparse.indices]
                if attr.HasField("g"):
                    graphs.append(attr.g)
                graphs.extend(attr.graphs)
    return nodes, tensors


def _check_signature(model, board_size):
    initializers = {init.name for init in model.graph.initializer}
    inputs = [i for i in model.graph.input if i.name not in initializers]
    outputs = list(model.graph.output)
    if len(inputs) != 1 or len(outputs) != 1:
        raise CandidateError("model must have exactly one input and one output")

    tensor_type = inputs[0].type.tensor_type
    if not inputs[0].type.HasField("tensor_type") or tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise CandidateError("input must be a float32 tensor")
    dims = tensor_type.shape.dim
    if not tensor_type.HasField("shape") or len(dims) != 4:
        raise CandidateError(f"input must have shape [B, 2, {board_size}, {board_size}]")
    if dims[0].HasField("dim_value"):
        raise CandidateError("batch dimension of the input must be dynamic")
    for dim, expected in zip(dims[1:], (2, board_size, board_size)):
        if dim.HasField("dim_value") and dim.dim_value != expected:
            raise CandidateError(f"input must have shape [B, 2, {board_size}, {board_size}]")


def _test_run(session, board_size, batch):
    board = np.zeros((batch, 2, board_size, board_size), dtype=np.float32)
    try:
        out = session.run(None, {session.get_inputs()[0].name: board})[0]
    except Exception:
        raise CandidateError("model failed on an empty-board test input") from None
    expected = (batch, board_size * board_size)
    if not isinstance(out, np.ndarray) or out.dtype != np.float32 or out.shape != expected:
        got = list(out.shape) if isinstance(out, np.ndarray) else type(out).__name__
        raise CandidateError(f"output must be float32 [B, {board_size * board_size}]; got {got} for B={batch}")
    if np.isnan(out).any():
        raise CandidateError("model output contains NaN on an empty board")


def load_session(model_bytes):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(model_bytes, sess_options=options, providers=["CPUExecutionProvider"])


def validate(model_bytes, board_size, max_bytes):
    """Run every upload check and return a ready ONNX Runtime session."""
    if len(model_bytes) > max_bytes:
        raise CandidateError("model exceeds the size limit")
    try:
        model = onnx.load_from_string(model_bytes)
    except Exception:
        raise CandidateError("not a valid ONNX model") from None

    # Structural checks run before the ONNX checker so nothing touches the filesystem on behalf of a
    # tensor that points at external data.
    nodes, tensors = _walk(model)
    if any(node.domain not in STANDARD_DOMAINS for node in nodes):
        raise CandidateError("custom operators are not allowed; use standard ONNX operators only")
    if any(t.data_location == onnx.TensorProto.EXTERNAL or len(t.external_data) for t in tensors):
        raise CandidateError("external data is not allowed; embed all weights in the model")

    try:
        onnx.checker.check_model(model)
    except Exception:
        raise CandidateError("ONNX checker rejected the model") from None
    _check_signature(model, board_size)

    try:
        session = load_session(model_bytes)
    except Exception:
        raise CandidateError("ONNX Runtime could not load the model") from None
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise CandidateError("model must have exactly one input and one output")
    # Two batch sizes catch a batch dimension that is fixed inside the graph.
    for batch in (1, 3):
        _test_run(session, board_size, batch)
    return session


def illegal_moves(inner):
    """Mask of cells the side to move may not play, from the borderless HexHex board `[B, 2, n, n]`.

    Mirrors the mask in `Conv.forward`: occupied cells are illegal, except that with exactly one stone and
    no swap marker (0.001) on the board the stone's cell means "swap". The marker sits next to a stone after
    a swap and on the first stone when swapping is disabled, so both cases block the swap.
    """
    occupied = (inner.sum(1) > 0).flatten(1)
    swap_ok = (inner.sum((1, 2, 3)) - 1).abs() < 1e-4
    return occupied & ~swap_ok[:, None]


class CandidatePlayer:
    """Callable player for `MultiHexGame`: HexHex board batch `[B, 2, n+2, n+2]` → logits `[B, n*n]`.

    The board arrives already rotated to the side-to-move's perspective (`Board.board_tensor`), so the
    wrapper only strips the border, drops the internal swap markers and blocks illegal cells. Illegal cells
    must be masked exactly: `Board.set_stone` answers an illegal move with `SystemExit`.
    """

    def __init__(self, session, board_size):
        self.session = session
        self.board_size = board_size
        self.input_name = session.get_inputs()[0].name

    def __call__(self, x):
        inner = x[:, :, 1:-1, 1:-1].detach().float().cpu()
        board = (inner >= 0.5).float().numpy()
        try:
            out = self.session.run(None, {self.input_name: board})[0]
            out = np.asarray(out, dtype=np.float32)
        except Exception:
            raise CandidateError("model failed during a game") from None
        if out.shape != (len(board), self.board_size ** 2):
            raise CandidateError(f"output must have shape [B, {self.board_size ** 2}] for every batch size")

        logits = torch.nan_to_num(torch.from_numpy(out), nan=-1e9, posinf=1e9, neginf=-1e9)
        logits[illegal_moves(inner)] = -float("inf")
        return logits.to(x.device)
