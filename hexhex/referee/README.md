# Hex referee

A local service that plays your 11×11 Hex model against fixed opponents and reports the results.

Base URL: `http://127.0.0.1:8765`

## Quick start

1. Export your model to ONNX following the contract below.
2. Check that the referee accepts it:
   `curl -s -H 'Content-Type: application/octet-stream' --data-binary @model.onnx http://127.0.0.1:8765/validate`
3. Play the `random` opponent first. A working model should win nearly every game; if it doesn't, check the
   orientation of your input and output.
4. Then evaluate against `hexhex_v1`.

```python
import time
import requests

BASE = "http://127.0.0.1:8765"
HEADERS = {"Content-Type": "application/octet-stream"}

with open("model.onnx", "rb") as f:
    model = f.read()

job = requests.post(f"{BASE}/evaluations", params={"opponent": "random"},
                    data=model, headers=HEADERS).json()
while True:
    record = requests.get(f"{BASE}/evaluations/{job['id']}").json()
    if record["status"] in ("done", "failed"):
        break
    time.sleep(5)
print(record)
```

## Model contract

- One input, float32, shape `[B, 2, 11, 11]`, where B is dynamic.
  - Channel 0 holds your stones, channel 1 your opponent's (1.0 = stone, 0.0 = empty).
  - The board is always oriented so that you connect index 0 to index 10 along the first spatial axis.
    When you actually connect the other way, the referee transposes the board for you.
- One output, float32, shape `[B, 121]`: a score per cell, higher is better. Index `k` is cell
  `(k // 11, k % 11)` in the same orientation as the input.
- The referee plays your highest-scoring legal cell. You don't need to mask occupied cells yourself.
- Swap rule: when exactly one stone is on the board, giving that stone's cell your highest score means
  "swap". When a swap isn't allowed, that cell is ignored.
- Standard ONNX operators only: no custom ops, no external data files, at most 100 MB. PyTorch's
  `torch.onnx.export` writes the weights to a separate `.data` file by default; pass `external_data=False`.
- The referee runs your model with ONNX Runtime on the CPU. Before accepting it, it runs the model on empty
  boards with B = 1 and B = 3 and checks the output shape and that it has no NaN.

### Which cells touch

Every cell touches six others. Cell `(i, j)` touches `(i-1, j)`, `(i+1, j)`, `(i, j-1)`, `(i, j+1)` and, of
the two diagonals, only `(i-1, j+1)` and `(i+1, j-1)`. The cells `(i-1, j-1)` and `(i+1, j+1)` don't touch
`(i, j)`. In the grid below, X marks the cells that touch O:

```
        j-1   j   j+1
  i-1    .    X    X
  i      X    O    X
  i+1    X    X    .
```

This is a hex board drawn as a rhombus, with each row shifted half a cell to the right of the row above. The
rule holds in the orientation you're given, whichever way you actually connect. For example, your stones on
`(0, 10), (1, 9), …, (10, 0)` connect index 0 to index 10 and win; stones on `(0, 0), (1, 1), …, (10, 10)` don't
form a chain.

## Match rules

- 11×11 Hex with the swap rule.
- Each evaluation is 242 games. Every game starts with one stone already placed: each of the 121 cells is
  used as the opening, twice, once from each side.
- Both players play greedily: the highest-scoring legal cell.
- `as_first` covers games where you own the pre-placed stone. `as_second` covers games where you move first
  after it, and may swap.
- Every evaluation plays the same openings, so evaluating the same model twice against `hexhex_v1` gives the
  same result. Against `random` the results vary.
- Evaluations run one at a time, in the order they were submitted.

## Endpoints

Every response is JSON. Errors have the form `{"error": "<short reason>"}`.

- `GET /info`: opponents and limits.
- `POST /validate`: body = ONNX bytes. Checks the model without playing. Returns `{"ok": true}`, or HTTP 400
  with `{"ok": false, "error": "..."}`.
- `POST /evaluations?opponent=<id>`: body = ONNX bytes. Validates the model like `/validate`, queues a match
  and returns HTTP 202 with `{"id": "...", "status": "queued", "queue_position": n}`, where `n` is the number
  of evaluations ahead of yours.
- `GET /evaluations/<id>`: job status and, when done, the results.
- `GET /evaluations`: all jobs, newest first.

Uploads are sent as `Content-Type: application/octet-stream` with a `Content-Length` header (HTTP 411
without it, 413 when larger than the limit).

A job record looks like this:

```json
{
  "id": "3f9c2a1e",
  "status": "done",
  "opponent": "hexhex_v1",
  "model_sha256": "…",
  "submitted_at": "2026-11-07T18:00:00Z",
  "started_at": "2026-11-07T18:00:01Z",
  "finished_at": "2026-11-07T18:03:10Z",
  "result": {
    "games": 242, "wins": 86, "losses": 156,
    "win_rate": 0.355, "ci95": [0.298, 0.418],
    "as_first":  {"wins": 48, "losses": 73},
    "as_second": {"wins": 38, "losses": 83}
  }
}
```

- `status` is `queued`, `running`, `done` or `failed`. A failed job has `"error": "<short reason>"` instead
  of `result`.
- `ci95` is a 95% Wilson interval for `win_rate`, about ±6 points at 242 games.
