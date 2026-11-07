#!/usr/bin/env python
"""Referee HTTP service: plays uploaded ONNX candidates against fixed opponents and returns only results.

    uv run python -m hexhex.referee.server

The agent-facing API is documented in hexhex/referee/README.md. What the agent learns is exactly what this
module returns, so responses carry aggregate results only: no moves, game records, scores, opponent
metadata or tracebacks (those go to the server log).
"""
import copy
import hashlib
import ipaddress
import json
import os
import queue
import secrets
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import hydra
from omegaconf import DictConfig

from hexhex.model.hexconvolution import RandomModel
from hexhex.referee import candidate
from hexhex.referee.match import games_per_evaluation, run_match
from hexhex.utils.logger import logger
from hexhex.utils.utils import load_model

ACTIVE = ("queued", "running")


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write(path, data):
    tmp = path.with_name(f"{path.name}.{secrets.token_hex(4)}.tmp")  # unique: concurrent uploads may share a path
    tmp.write_bytes(data)
    os.replace(tmp, path)


def load_opponents(cfg):
    """Opponent id → player, loaded once at startup. Paths stay in the config and are never returned."""
    opponents = {}
    for name, spec in cfg.opponents.items():
        if spec != "random" and not Path(spec).is_file():
            raise FileNotFoundError(f"opponent {name}: no checkpoint at {spec}; copy one there or override "
                                    f"opponents.{name}=<path>")
        model = RandomModel(cfg.board_size) if spec == "random" else load_model(spec)
        if model.board_size != cfg.board_size:
            raise ValueError(f"opponent {name} is {model.board_size}x{model.board_size}, expected {cfg.board_size}")
        opponents[name] = model
    return opponents


class Referee:
    """Job queue and persistence. Request threads call validate/submit/get/list; one worker runs the matches."""

    def __init__(self, cfg, opponents):
        self.cfg = cfg
        self.board_size = cfg.board_size
        self.max_bytes = int(cfg.max_model_mb * 1024 * 1024)
        self.opponents = opponents
        data_dir = Path(cfg.data_dir).resolve()
        self.submissions_dir = data_dir / "submissions"
        self.evaluations_dir = data_dir / "evaluations"
        self.submissions_dir.mkdir(parents=True, exist_ok=True)
        self.evaluations_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs = {}  # id → record, oldest first
        self._queue = queue.Queue()
        self._worker = None
        self._load_jobs()

    def _load_jobs(self):
        records = []
        for path in self.evaluations_dir.glob("*.json"):
            try:
                records.append(json.loads(path.read_text()))
            except (OSError, ValueError):
                logger.exception(f"skipping unreadable job record {path}")
        for record in sorted(records, key=lambda r: (r["submitted_at"], r["id"])):
            if record["status"] in ACTIVE:
                record.pop("result", None)
                record.update(status="failed", error="server restarted", finished_at=_now())
                self._save(record)
            self._jobs[record["id"]] = record
        logger.info(f"loaded {len(records)} job records from {self.evaluations_dir}")

    def _save(self, record):
        _atomic_write(self.evaluations_dir / f"{record['id']}.json", json.dumps(record, indent=2).encode())

    def info(self):
        n = self.board_size
        return {
            "board_size": n,
            "opponents": sorted(self.opponents),
            "games_per_evaluation": games_per_evaluation(self.cfg.match, n),
            "max_model_mb": self.cfg.max_model_mb,
            "input_shape": ["B", 2, n, n],
            "output_shape": ["B", n * n],
        }

    def validate(self, model_bytes):
        return candidate.validate(model_bytes, self.board_size, self.max_bytes)

    def submit(self, model_bytes, opponent):
        """Validate, store and queue a candidate. Returns (record, number of jobs ahead of it)."""
        self.validate(model_bytes)
        sha = hashlib.sha256(model_bytes).hexdigest()
        path = self.submissions_dir / f"{sha}.onnx"
        if not path.exists():
            _atomic_write(path, model_bytes)
        with self._lock:
            job_id = secrets.token_hex(4)
            while job_id in self._jobs:
                job_id = secrets.token_hex(4)
            ahead = sum(1 for r in self._jobs.values() if r["status"] in ACTIVE)
            record = {
                "id": job_id,
                "status": "queued",
                "opponent": opponent,
                "model_sha256": sha,
                "submitted_at": _now(),
                "started_at": None,
                "finished_at": None,
            }
            self._jobs[job_id] = record
            self._save(record)
            self._queue.put(job_id)
        logger.info(f"queued evaluation {job_id}: model {sha[:12]} vs {opponent} ({ahead} ahead)")
        return copy.deepcopy(record), ahead

    def get(self, job_id):
        with self._lock:
            record = self._jobs.get(job_id)
            return copy.deepcopy(record) if record is not None else None

    def list(self):
        with self._lock:
            return [copy.deepcopy(r) for r in reversed(self._jobs.values())]

    def start(self):
        self._worker = threading.Thread(target=self._work, name="referee-worker", daemon=True)
        self._worker.start()

    def stop(self, timeout=None):
        self._queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout)

    def _work(self):
        while (job_id := self._queue.get()) is not None:
            self._run(job_id)

    def _run(self, job_id):
        with self._lock:
            record = self._jobs[job_id]
            record.update(status="running", started_at=_now())
            self._save(record)
        logger.info(f"running evaluation {job_id}")
        # BaseException, not Exception: an illegal move makes Board.set_stone raise SystemExit, which would
        # otherwise end the worker thread silently and leave the job "running" forever.
        try:
            model_bytes = (self.submissions_dir / f"{record['model_sha256']}.onnx").read_bytes()
            player = candidate.CandidatePlayer(candidate.load_session(model_bytes), self.board_size)
            result = run_match(player, self.opponents[record["opponent"]], self.cfg.match, self.cfg.get("seed"))
            update = {"status": "done", "result": result}
            logger.info(f"evaluation {job_id} done: {result}")
        except candidate.CandidateError as e:
            logger.warning(f"evaluation {job_id} failed: {e}")
            update = {"status": "failed", "error": str(e)}
        except BaseException:
            logger.exception(f"evaluation {job_id} failed with an internal error")
            update = {"status": "failed", "error": "internal error"}
        with self._lock:
            record.update(update, finished_at=_now())
            self._save(record)


class Handler(BaseHTTPRequestHandler):
    server_version = "HexReferee/1"
    sys_version = ""
    timeout = 120  # seconds; bounds a stalled upload

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_PATCH(self):
        self._dispatch("PATCH")

    def do_DELETE(self):
        self._dispatch("DELETE")

    @property
    def referee(self):
        return self.server.referee

    def _routes(self, path):
        """Map a path to {method: handler}, or None for an unknown path."""
        if path == "/info":
            return {"GET": self._info}
        if path == "/validate":
            return {"POST": self._validate}
        if path == "/evaluations":
            return {"GET": self._list, "POST": self._submit}
        job_id = path.removeprefix("/evaluations/")
        if job_id != path and job_id and "/" not in job_id:
            return {"GET": lambda: self._get(job_id)}
        return None

    def _dispatch(self, method):
        self._responded = False
        try:
            url = urlsplit(self.path)
            self._query = parse_qs(url.query)
            routes = self._routes(url.path)
            if routes is None:
                return self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            if method not in routes:
                return self._send_json(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "method not allowed"},
                                       {"Allow": ", ".join(routes)})
            routes[method]()
        except TimeoutError:
            raise  # BaseHTTPRequestHandler drops the connection
        except Exception:
            logger.exception(f"{method} {self.path} failed")
            if not self._responded:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal error"})

    def _send_json(self, status, body, headers=None):
        data = json.dumps(body).encode()
        self._responded = True
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def send_error(self, code, message=None, explain=None):
        # The stdlib calls this for malformed requests and unsupported methods; keep those JSON too and
        # never echo the request back.
        self.close_connection = True
        self._send_json(code, {"error": HTTPStatus(code).phrase.lower()})

    def log_message(self, format, *args):
        logger.info(f"{self.address_string()} {format % args}")

    def _read_body(self):
        """Return the uploaded bytes, or None after an error response has been sent."""
        length = self.headers.get("Content-Length")
        if length is None:
            self._send_json(HTTPStatus.LENGTH_REQUIRED, {"error": "Content-Length required"})
            return None
        if not length.strip().isdigit():
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid Content-Length"})
            return None
        length = int(length)
        if length > self.referee.max_bytes:
            self.close_connection = True  # the body stays unread
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                            {"error": f"model exceeds {self.referee.cfg.max_model_mb} MB"})
            return None
        body = self.rfile.read(length)
        if len(body) != length:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "incomplete upload"})
            return None
        return body

    def _info(self):
        self._send_json(HTTPStatus.OK, self.referee.info())

    def _validate(self):
        body = self._read_body()
        if body is None:
            return
        try:
            self.referee.validate(body)
        except candidate.CandidateError as e:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(e)})
        self._send_json(HTTPStatus.OK, {"ok": True})

    def _submit(self):
        body = self._read_body()
        if body is None:
            return
        opponent = self._query.get("opponent", [None])[0]
        if opponent is None:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "opponent is required"})
        if opponent not in self.referee.opponents:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "unknown opponent; see GET /info"})
        try:
            record, ahead = self.referee.submit(body, opponent)
        except candidate.CandidateError as e:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
        self._send_json(HTTPStatus.ACCEPTED, {"id": record["id"], "status": "queued", "queue_position": ahead})

    def _get(self, job_id):
        record = self.referee.get(job_id)
        if record is None:
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown evaluation id"})
        self._send_json(HTTPStatus.OK, record)

    def _list(self):
        self._send_json(HTTPStatus.OK, self.referee.list())


def make_server(referee, host, port):
    server = ThreadingHTTPServer((host, port), Handler)
    server.referee = referee
    return server


def _is_loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@hydra.main(version_base=None, config_path="../../conf", config_name="referee")
def main(cfg: DictConfig):
    if not _is_loopback(cfg.host):
        raise ValueError(f"host must be a loopback address, got {cfg.host}")
    referee = Referee(cfg, load_opponents(cfg))
    referee.start()
    server = make_server(referee, cfg.host, cfg.port)
    host, port = server.server_address[:2]
    logger.info(f"referee listening on http://{host}:{port} (opponents: {', '.join(sorted(referee.opponents))})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down; unfinished jobs are marked failed on the next start")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
