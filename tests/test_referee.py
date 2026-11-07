import copy
import http.client
import json
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from onnx import TensorProto, helper, numpy_helper

from hexhex.creation.create_model import create_model
from hexhex.logic.hexboard import Board, to_move
from hexhex.model.hexconvolution import RandomModel
from hexhex.referee import candidate
from hexhex.referee import server as referee_server
from hexhex.referee.candidate import CandidateError, CandidatePlayer, illegal_moves
from hexhex.referee.export_candidate import export_onnx
from hexhex.referee.match import run_match, wilson_interval
from hexhex.utils import utils
from hexhex.utils.utils import load_cpu_model

N = 11
ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = 100 * 2**20


# --- ONNX fixtures built with onnx.helper -------------------------------------------------------------

AXES = numpy_helper.from_array(np.array([1], dtype=np.int64), "axes")


def _onnx_bytes(nodes, input_shape=("B", 2, N, N), output_shape=("B", N * N), initializers=(), extra_opsets=()):
    graph = helper.make_graph(
        nodes, "candidate",
        [helper.make_tensor_value_info("board", TensorProto.FLOAT, list(input_shape))],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, list(output_shape))],
        initializer=[AXES, *initializers],
    )
    opsets = [helper.make_opsetid("", 17), *(helper.make_opsetid(d, 1) for d in extra_opsets)]
    model = helper.make_model(graph, opset_imports=opsets)
    model.ir_version = 8
    return model.SerializeToString()


def _stone_count(out):
    """board [B, 2, n, n] → stones per cell, flattened to [B, n*n] under the name `out`."""
    return [helper.make_node("ReduceSum", ["board", "axes"], ["count"], keepdims=0),
            helper.make_node("Flatten", ["count"], [out], axis=1)]


def prefer_empty_model():
    """−Flatten(ReduceSum(board, axis=1)): empty cells score 0, occupied cells −1."""
    return _onnx_bytes(_stone_count("flat") + [helper.make_node("Neg", ["flat"], ["logits"])])


def prefer_occupied_model():
    """Flatten(ReduceSum(board, axis=1)): occupied cells get the highest scores."""
    return _onnx_bytes(_stone_count("logits"))


def _external_tensor(name):
    tensor = helper.make_tensor(name, TensorProto.FLOAT, [1], [0.0])
    tensor.ClearField("float_data")
    tensor.data_location = TensorProto.EXTERNAL
    entry = tensor.external_data.add()
    entry.key, entry.value = "location", "weights.bin"
    return tensor


def _custom_op_in_subgraph():
    branch = helper.make_graph(
        [helper.make_node("Foo", ["flat"], ["out"], domain="com.example")], "branch", [],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["B", N * N])])
    cond = numpy_helper.from_array(np.array(True), "cond")
    nodes = _stone_count("flat") + [
        helper.make_node("If", ["cond"], ["logits"], then_branch=branch, else_branch=branch)]
    return _onnx_bytes(nodes, initializers=[cond], extra_opsets=["com.example"])


INVALID_MODELS = {
    "garbage": lambda: b"definitely not an onnx model" * 10,
    "input 13x13": lambda: _onnx_bytes(_stone_count("logits"), input_shape=("B", 2, 13, 13),
                                       output_shape=("B", 169)),
    "output [B, 11, 11]": lambda: _onnx_bytes(
        [helper.make_node("ReduceSum", ["board", "axes"], ["count"], keepdims=0),
         helper.make_node("Neg", ["count"], ["logits"])], output_shape=("B", N, N)),
    "fixed batch input": lambda: _onnx_bytes(_stone_count("logits"), input_shape=(1, 2, N, N),
                                             output_shape=(1, N * N)),
    "fixed batch inside graph": lambda: _onnx_bytes(
        [helper.make_node("ReduceSum", ["board", "axes"], ["count"], keepdims=0),
         helper.make_node("Reshape", ["count", "shape"], ["logits"])],
        initializers=[numpy_helper.from_array(np.array([1, N * N], dtype=np.int64), "shape")]),
    "custom domain": lambda: _onnx_bytes([helper.make_node("Foo", ["board"], ["logits"], domain="com.example")],
                                         extra_opsets=["com.example"]),
    "custom domain in If branch": _custom_op_in_subgraph,
    "external initializer": lambda: _onnx_bytes(
        _stone_count("flat") + [helper.make_node("Add", ["flat", "w"], ["logits"])],
        initializers=[_external_tensor("w")]),
    "external Constant attribute": lambda: _onnx_bytes(
        _stone_count("flat") + [helper.make_node("Constant", [], ["w"], value=_external_tensor("c")),
                                helper.make_node("Add", ["flat", "w"], ["logits"])]),
    "NaN on empty board": lambda: _onnx_bytes(
        _stone_count("flat") + [helper.make_node("Add", ["flat", "nan"], ["logits"])],
        initializers=[numpy_helper.from_array(np.array([np.nan], dtype=np.float32), "nan")]),
}


# --- shared helpers -----------------------------------------------------------------------------------

def _base_cfg(**overrides):
    cfg = OmegaConf.load(ROOT / "conf" / "referee.yaml")
    cfg.match.num_games = 4
    cfg.match.batch_size = 4
    return OmegaConf.merge(cfg, overrides)


def _small_model(seed=0):
    model_cfg = OmegaConf.load(ROOT / "conf" / "preset" / "11x11.yaml").model
    model_cfg.layers = 1
    model_cfg.intermediate_channels = 8
    torch.manual_seed(seed)
    return create_model(model_cfg).eval(), OmegaConf.to_container(model_cfg)


def _player(model_bytes):
    return CandidatePlayer(candidate.validate(model_bytes, N, MAX_BYTES), N)


class FakeSession:
    """Stands in for an ONNX Runtime session: records every input and returns fixed scores."""

    def __init__(self, scores=None):
        self.inputs = []
        self.scores = np.zeros(N * N, dtype=np.float32) if scores is None else scores

    def get_inputs(self):
        return [SimpleNamespace(name="board")]

    def run(self, output_names, feed):
        board = feed["board"]
        self.inputs.append(board.copy())
        return [np.tile(self.scores.astype(np.float32), (len(board), 1))]


def _random_game_states(num_games, rng, swap_prob=0.5, switch_allowed=True):
    """Yield (board_tensor, player, legal_moves) before every move of uniformly random games."""
    for _ in range(num_games):
        board = Board(N, switch_allowed=switch_allowed)
        while not board.winner:
            yield board.board_tensor.clone(), board.player, set(board.legal_moves)
            if len(board.made_moves) == 1 and board.made_moves <= board.legal_moves:
                [stone] = board.made_moves
                move = stone if rng.random() < swap_prob else rng.choice(sorted(board.legal_moves - {stone}))
            else:
                move = rng.choice(sorted(board.legal_moves))
            board.set_stone(move)


# --- validation ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", INVALID_MODELS)
def test_validate_rejects(name):
    with pytest.raises(CandidateError):
        candidate.validate(INVALID_MODELS[name](), N, MAX_BYTES)


@pytest.mark.parametrize("name, reason", [
    ("custom domain", "custom operators"),
    ("custom domain in If branch", "custom operators"),
    ("external initializer", "external data"),
    ("external Constant attribute", "external data"),
    ("fixed batch input", "batch dimension"),
    ("fixed batch inside graph", "test input"),
    ("output [B, 11, 11]", "output must be"),
    ("NaN on empty board", "NaN"),
])
def test_validate_reason(name, reason):
    with pytest.raises(CandidateError, match=reason):
        candidate.validate(INVALID_MODELS[name](), N, MAX_BYTES)


def test_validate_rejects_over_size_limit():
    model = prefer_empty_model()
    with pytest.raises(CandidateError, match="size limit"):
        candidate.validate(model, N, len(model) - 1)
    candidate.validate(model, N, len(model))


def test_validate_accepts_minimal_model():
    session = candidate.validate(prefer_empty_model(), N, MAX_BYTES)
    out = session.run(None, {"board": np.zeros((2, 2, N, N), np.float32)})[0]
    assert out.shape == (2, N * N)


# --- wrapper ------------------------------------------------------------------------------------------

def _seen_by_candidate(board):
    session = FakeSession()
    CandidatePlayer(session, N)(board.board_tensor.unsqueeze(0))
    return session.inputs[-1][0]


def test_wrapper_input_is_stones_from_side_to_move():
    board = Board(N)
    board.set_stone((2, 3))  # player 0
    expected = np.zeros((2, N, N), np.float32)
    expected[1, 3, 2] = 1  # player 1 to move: board transposed, opponent in channel 1
    np.testing.assert_array_equal(_seen_by_candidate(board), expected)

    board.set_stone((4, 6))  # player 1
    expected = np.zeros((2, N, N), np.float32)
    expected[0, 2, 3] = 1
    expected[1, 4, 6] = 1
    np.testing.assert_array_equal(_seen_by_candidate(board), expected)

    board.set_stone((7, 1))  # player 0
    expected = np.zeros((2, N, N), np.float32)
    expected[0, 6, 4] = 1
    expected[1, 3, 2] = 1
    expected[1, 1, 7] = 1
    np.testing.assert_array_equal(_seen_by_candidate(board), expected)


@pytest.mark.parametrize("switch_allowed", [True, False])
def test_wrapper_input_has_no_swap_markers(switch_allowed):
    board = Board(N, switch_allowed=switch_allowed)
    board.set_stone((2, 3))
    if switch_allowed:
        board.set_stone((2, 3))  # swap: HexHex writes a 0.001 marker
    assert (board.board_tensor == 0.001).any()
    expected = np.zeros((2, N, N), np.float32)
    expected[1, 3, 2] = 1
    np.testing.assert_array_equal(_seen_by_candidate(board), expected)


def _illegal(board):
    return illegal_moves(board.board_tensor[None, :, 1:-1, 1:-1])[0]


def test_swap_allowed_only_with_one_stone_and_no_marker():
    stone_idx = 3 * N + 2  # (2, 3) seen transposed by player 1

    board = Board(N)
    board.set_stone((2, 3))
    assert not _illegal(board).any()

    swapped = Board(N)
    swapped.set_stone((2, 3))
    swapped.set_stone((2, 3))
    assert _illegal(swapped).nonzero().flatten().tolist() == [stone_idx]

    no_switch = Board(N, switch_allowed=False)
    no_switch.set_stone((2, 3))
    assert _illegal(no_switch).nonzero().flatten().tolist() == [stone_idx]

    board.set_stone((4, 6))
    assert _illegal(board).sum() == 2


@pytest.mark.parametrize("switch_allowed", [True, False])
def test_mask_matches_board_legality(switch_allowed):
    states = list(_random_game_states(30, random.Random(0), switch_allowed=switch_allowed))
    illegal = illegal_moves(torch.stack([t for t, _, _ in states])[:, :, 1:-1, 1:-1])
    for (_, player, legal_moves), row in zip(states, illegal):
        allowed = {to_move(utils.correct_position1d(k, N, player), N) for k in (~row).nonzero().flatten().tolist()}
        assert allowed == legal_moves


def test_candidate_preferring_occupied_cells_plays_legal_games():
    result = run_match(_player(prefer_occupied_model()), RandomModel(N), _base_cfg(match={"num_games": 20}).match)
    assert result["games"] == 20


@pytest.mark.parametrize("fill", ["nan", "inf", "mixed"])
def test_wrapper_survives_non_finite_outputs(fill):
    scores = np.full(N * N, {"nan": np.nan, "inf": np.inf, "mixed": np.nan}[fill], dtype=np.float32)
    if fill == "mixed":
        scores[::3] = np.inf
        scores[1::3] = -np.inf
    player = CandidatePlayer(FakeSession(scores), N)
    result = run_match(player, RandomModel(N), _base_cfg(match={"num_games": 10}).match)
    assert result["games"] == 10


def test_wrapper_rejects_wrong_output_shape_during_game():
    session = FakeSession(np.zeros(N * N - 1, dtype=np.float32))
    with pytest.raises(CandidateError, match="shape"):
        CandidatePlayer(session, N)(Board(N).board_tensor.unsqueeze(0))


# --- match --------------------------------------------------------------------------------------------

def test_wilson_interval():
    low, high = wilson_interval(71, 200)
    assert (round(low, 3), round(high, 3)) == (0.292, 0.423)


def test_match_counts_are_consistent():
    result = run_match(_player(prefer_empty_model()), RandomModel(N), _base_cfg().match)
    assert result["wins"] + result["losses"] == result["games"] == 4
    for side in ("as_first", "as_second"):
        assert result[side]["wins"] + result[side]["losses"] == 2
    assert result["as_first"]["wins"] + result["as_second"]["wins"] == result["wins"]
    assert result["ci95"][0] <= result["win_rate"] <= result["ci95"][1]


def test_mirror_match_is_even(tmp_path):
    model, _ = _small_model()
    export_onnx(model, tmp_path / "mirror.onnx")
    reference = copy.deepcopy(model).to(utils.device)
    result = run_match(_player((tmp_path / "mirror.onnx").read_bytes()), reference,
                       _base_cfg(match={"num_games": 40, "batch_size": 40}).match)
    assert abs(result["win_rate"] - 0.5) <= 0.1


@pytest.fixture(scope="module")
def trained_onnx(tmp_path_factory):
    path = tmp_path_factory.mktemp("trained") / "11_2w4_2000.onnx"
    export_onnx(load_cpu_model(ROOT / "reference_models" / "11_2w4_2000.pt"), path)
    return path.read_bytes()


def test_trained_model_beats_random(trained_onnx):
    result = run_match(_player(trained_onnx), RandomModel(N),
                       _base_cfg(match={"num_games": 40, "batch_size": 40}).match)
    assert result["win_rate"] >= 0.95


def test_adapter_parity_with_torch_reference(trained_onnx):
    reference = load_cpu_model(ROOT / "reference_models" / "11_2w4_2000.pt")
    rng = random.Random(1)
    # No swaps: right after a swap the contract input lacks HexHex's 0.001 marker, so moves may differ there.
    states = [t for t, _, _ in _random_game_states(40, rng, swap_prob=0.0)]
    x = torch.stack(rng.sample(states, 500))
    with torch.no_grad():
        torch_logits = reference(x)
    onnx_choice = _player(trained_onnx)(x).argmax(1)
    torch_choice = torch_logits.argmax(1)
    for i in (onnx_choice != torch_choice).nonzero().flatten().tolist():  # only near-ties may differ
        assert torch_logits[i, torch_choice[i]] - torch_logits[i, onnx_choice[i]] < 1e-3


# --- server -------------------------------------------------------------------------------------------

@pytest.fixture
def ref_checkpoint(tmp_path):
    model, model_cfg = _small_model()
    path = tmp_path / "secret_reference.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": model_cfg}, path)
    return path


@pytest.fixture
def referee(tmp_path, ref_checkpoint):
    cfg = _base_cfg(data_dir=str(tmp_path / "referee_data"), opponents={"hexhex_v1": str(ref_checkpoint)})
    ref = referee_server.Referee(cfg, referee_server.load_opponents(cfg))
    ref.start()
    http_server = referee_server.make_server(ref, "127.0.0.1", 0)
    thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    thread.start()
    ref.base_url = f"http://127.0.0.1:{http_server.server_address[1]}"
    yield ref
    http_server.shutdown()
    http_server.server_close()
    ref.stop(timeout=30)


def _call(base_url, method, path, body=None):
    """Return (status, raw body). Asserts every response is JSON without a traceback."""
    headers = {"Content-Type": "application/octet-stream"} if body is not None else {}
    request = urllib.request.Request(base_url + path, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status, raw = response.status, response.read().decode()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read().decode()
    json.loads(raw)
    assert "Traceback" not in raw
    return status, raw


def _wait_for(base_url, job_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        _, raw = _call(base_url, "GET", f"/evaluations/{job_id}")
        record = json.loads(raw)
        if record["status"] in ("done", "failed"):
            return record
        time.sleep(0.1)
    raise AssertionError(f"evaluation {job_id} did not finish")


def test_server_info_lists_opponents_without_paths(referee, ref_checkpoint):
    status, raw = _call(referee.base_url, "GET", "/info")
    assert status == 200
    info = json.loads(raw)
    assert info["opponents"] == ["hexhex_v1", "random"]
    assert info["board_size"] == N
    assert info["games_per_evaluation"] == 4
    assert info["input_shape"] == ["B", 2, N, N] and info["output_shape"] == ["B", N * N]
    assert ref_checkpoint.name not in raw and ".pt" not in raw


def test_server_validate(referee):
    status, raw = _call(referee.base_url, "POST", "/validate", prefer_empty_model())
    assert (status, json.loads(raw)) == (200, {"ok": True})
    status, raw = _call(referee.base_url, "POST", "/validate", INVALID_MODELS["custom domain"]())
    assert status == 400
    assert json.loads(raw)["ok"] is False and json.loads(raw)["error"]


@pytest.mark.parametrize("opponent", ["random", "hexhex_v1"])
def test_server_evaluation_reaches_done(referee, opponent):
    status, raw = _call(referee.base_url, "POST", f"/evaluations?opponent={opponent}", prefer_empty_model())
    assert status == 202
    job = json.loads(raw)
    assert job["status"] == "queued" and job["queue_position"] == 0

    record = _wait_for(referee.base_url, job["id"])
    assert record["status"] == "done", record
    assert record["opponent"] == opponent
    result = record["result"]
    assert result["wins"] + result["losses"] == result["games"] == 4
    assert result["as_first"]["wins"] + result["as_second"]["wins"] == result["wins"]
    assert set(record) == {"id", "status", "opponent", "model_sha256", "submitted_at", "started_at",
                           "finished_at", "result"}

    stored = referee.submissions_dir / f"{record['model_sha256']}.onnx"
    assert stored.read_bytes() == prefer_empty_model()


def test_server_lists_newest_first(referee):
    ids = [json.loads(_call(referee.base_url, "POST", "/evaluations?opponent=random", prefer_empty_model())[1])["id"]
           for _ in range(2)]
    for job_id in ids:
        _wait_for(referee.base_url, job_id)
    status, raw = _call(referee.base_url, "GET", "/evaluations")
    assert status == 200
    assert [r["id"] for r in json.loads(raw)] == ids[::-1]


def test_server_rejects_invalid_submission_without_queueing(referee):
    status, _ = _call(referee.base_url, "POST", "/evaluations?opponent=random", INVALID_MODELS["garbage"]())
    assert status == 400
    assert json.loads(_call(referee.base_url, "GET", "/evaluations")[1]) == []


@pytest.mark.parametrize("method, path, body, expected", [
    ("POST", "/evaluations?opponent=nope", b"x", 400),
    ("POST", "/evaluations", b"x", 400),
    ("GET", "/evaluations/doesnotexist", None, 404),
    ("GET", "/nothing", None, 404),
    ("GET", "/validate", None, 405),
    ("DELETE", "/info", None, 405),
    ("POST", "/info", b"x", 405),
])
def test_server_error_statuses(referee, method, path, body, expected):
    status, raw = _call(referee.base_url, method, path, body)
    assert status == expected
    assert set(json.loads(raw)) == {"error"}


def _raw_post(referee, headers):
    host, port = referee.base_url.removeprefix("http://").split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=10)
    conn.putrequest("POST", "/validate")
    for key, value in headers.items():
        conn.putheader(key, value)
    conn.endheaders()
    response = conn.getresponse()
    raw = response.read().decode()
    conn.close()
    assert "Traceback" not in raw
    return response.status, json.loads(raw)


def test_server_requires_content_length(referee):
    status, body = _raw_post(referee, {})
    assert status == 411 and set(body) == {"error"}


def test_server_rejects_oversized_upload(referee):
    status, body = _raw_post(referee, {"Content-Length": str(referee.max_bytes + 1)})
    assert status == 413 and set(body) == {"error"}


def test_server_internal_error_has_no_traceback(referee, monkeypatch):
    def broken():
        raise RuntimeError("secret detail")
    monkeypatch.setattr(referee, "info", broken)
    status, raw = _call(referee.base_url, "GET", "/info")
    assert (status, json.loads(raw)) == (500, {"error": "internal error"})


def test_worker_survives_system_exit(referee, monkeypatch):
    real_run_match = referee_server.run_match

    def illegal_move(*args, **kwargs):
        raise SystemExit  # what Board.set_stone does on an illegal move
    monkeypatch.setattr(referee_server, "run_match", illegal_move)
    job = json.loads(_call(referee.base_url, "POST", "/evaluations?opponent=random", prefer_empty_model())[1])
    record = _wait_for(referee.base_url, job["id"])
    assert (record["status"], record["error"]) == ("failed", "internal error")
    assert "result" not in record

    monkeypatch.setattr(referee_server, "run_match", real_run_match)
    job = json.loads(_call(referee.base_url, "POST", "/evaluations?opponent=random", prefer_empty_model())[1])
    assert _wait_for(referee.base_url, job["id"])["status"] == "done"


def test_restart_marks_unfinished_jobs_failed(tmp_path):
    cfg = _base_cfg(data_dir=str(tmp_path / "referee_data"))
    evaluations = tmp_path / "referee_data" / "evaluations"
    evaluations.mkdir(parents=True)
    base = {"opponent": "random", "model_sha256": "0" * 64, "started_at": None, "finished_at": None}
    records = [
        {"id": "aaaa0001", "status": "done", "submitted_at": "2026-11-07T18:00:00Z",
         "result": {"games": 4}, **base},
        {"id": "aaaa0002", "status": "running", "submitted_at": "2026-11-07T18:01:00Z", **base},
        {"id": "aaaa0003", "status": "queued", "submitted_at": "2026-11-07T18:02:00Z", **base},
    ]
    for record in records:
        (evaluations / f"{record['id']}.json").write_text(json.dumps(record))

    ref = referee_server.Referee(cfg, {"random": RandomModel(N)})
    assert [r["id"] for r in ref.list()] == ["aaaa0003", "aaaa0002", "aaaa0001"]
    assert ref.get("aaaa0001")["status"] == "done"
    for job_id in ("aaaa0002", "aaaa0003"):
        record = ref.get(job_id)
        assert (record["status"], record["error"]) == ("failed", "server restarted")
        assert json.loads((evaluations / f"{job_id}.json").read_text())["status"] == "failed"
