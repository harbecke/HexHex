# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HexHex is a reinforcement learning agent for the board game Hex, trained via self-play without MCTS (unlike AlphaGo Zero). The project has two components:
- **Python backend** (`hexhex/`): RL training pipeline, game logic, neural network
- **React frontend** (`app/`): Browser-based UI using ONNX.js for in-browser inference
- **E2E tests** (`tests/app_e2e/`): Playwright smoke tests for the frontend

Live demo: https://cleeff.github.io/hex/

## Commands

### Python (Backend)
```bash
uv sync                                                     # Install dependencies
uv run pytest                                               # Run all tests
uv run pytest tests/test_file.py::test_name                # Run a single test
uv run python -m hexhex.training.repeated_self_training    # Run training loop
uv run python -m hexhex.interactive.interactive             # Launch interactive GUI
uv run tensorboard --logdir runs/                           # Visualize training
uv run python -m hexhex.solver.solve --size 3 --out tables/3x3.bin  # Solve small board
```

### JavaScript (Frontend — app/)
```bash
npm run dev      # Dev server (http://localhost:5173)
npm run build    # Production build → app/build/
npm test         # Run Vitest unit tests
npm run preview  # Serve production build locally (http://localhost:4173)
```

### Playwright E2E (root — run by human, not Claude)
```bash
npm install                  # Install Playwright and deps (run once)
npx playwright install       # Install browser binaries (run once)
npx playwright test          # Run smoke tests (starts vite preview automatically)
```

Tests live in `tests/app_e2e/`. Screenshots written to `tests/app_e2e/screenshots/` (gitignored). Requires a production build (`npm run build` inside `app/`) first.

### Linting
Ruff is configured in `pyproject.toml` with line-length 120. No explicit lint command is defined; use `uv run ruff check .` if needed.

## Architecture

### Data flow
1. Self-play games → training data (board state → best move)
2. CNN trained on that data
3. New model evaluated against prior versions via ELO ranking
4. Best model exported to ONNX → deployed in the React frontend

### Key components

**`hexhex/logic/`** — Game rules
- `hexboard.py`: board representation, move legality, Pie rule (switch rule)
- `hexgame.py`: batched multi-game runner for efficient self-play data generation

**`hexhex/model/hexconvolution.py`** — CNN architecture
- 18 conv layers, 64 channels, batch norm, skip connections
- Single output head (move logits) — no separate policy/value heads
- Wrapper classes for rotation invariance and switch rule handling

**`hexhex/creation/`** — Data and model factory
- `create_model.py`: instantiates models from Hydra/OmegaConf config
- `create_data.py`: generates batches of self-play games
- `noise.py`: exploration noise for move selection

**`hexhex/training/`** — Training pipeline
- `train.py`: core training loop
- `repeated_self_training.py`: iterative self-play → train → evaluate loop

**`hexhex/evaluation/` + `hexhex/elo/`** — Model ranking
- `evaluate_two_models.py`: head-to-head match
- `elo.py`: Bradley-Terry model for multi-agent ranking

**`hexhex/export/onnx_export.py`** — Converts trained PyTorch model to ONNX for web deployment

**`hexhex/solver/`** — Exact ground-truth solver for small boards (used to produce reference tables for training metrics)
- `encoding.py`: base-3 position keys, neighbour bitmasks, win detection
- `solve.py`: negamax with transposition table (no pie rule, no alpha-beta — explores every legal move from every position so the table covers the full reachable state space). CLI writes a binary table.
- `table.py`: `SolutionTable` loader; `winning(key) -> bool` returns "side-to-move wins" for any key in the table.
- Tables are stored as `tables/NxN.bin` (gitignored, regenerable). 3×3 solves in ~30 ms (44 KB), 4×4 in ~3 min (72 MB). 5×5 is not feasible in pure Python (~50–150 GB and tens of hours).

**`app/src/`** — React frontend (Vite + React 19 + TypeScript)
- `game/`: pure game logic (coords, rules, encoding) — fully unit-tested with Vitest
- `ai/modelWorker.ts`: Web Worker that loads the ONNX model via `onnxruntime-web` and runs inference
- `hooks/useAI.ts`: manages worker lifecycle, drives AI turn after player moves
- `components/HexBoard.tsx`: plain SVG hex grid renderer (no third-party grid library)
- `App.tsx`: `useReducer`-based game state, wires board + AI + controls together

### Frontend: hex geometry

Hexagons are **pointy-top** with circumradius 1. Cell centers are computed by `hexCenter(x, y)` in `HexBoard.tsx`:

```
cx = (x + 0.5) * √3  +  y * (√3 / 2)   // columns spaced √3 apart, rows sheared right
cy = (y + 0.5) * 1.5                      // rows spaced 1.5 apart
```

The shear (`y * √3/2`) is what gives the board its parallelogram shape — each row is offset half a hex-width to the right relative to the row above. Coordinates are in SVG user units where circumradius = 1; the SVG `viewBox` is computed dynamically from all hex centers plus their radii.

Cell ids use `posToId(x, y) = x + BOARD_SIZE * y` (row-major with x = column, y = row). Board origin is top-left; x increases right, y increases down.

### Frontend: board encoding for ONNX inference

The model always runs from the perspective of the **current agent as red** (trained with red going top↔bottom). The encoding in `game/encoding.ts` handles the perspective flip:

- **Agent is red** (`agentIsBlue = false`): loop is **y-major** (outer = y, inner = x). Channel 0 = agent (red) stones + red borders lit; channel 1 = opponent (blue) stones + blue borders lit. Model output element `k` → cell `posToId(k % 11, k ÷ 11)`. No re-indexing needed.

- **Agent is blue** (`agentIsBlue = true`): loop is **x-major** (outer = x, inner = y), which transposes the board so the blue agent's left↔right connectivity maps to the model's top↔bottom. Channel 0 = agent (blue) stones + blue borders lit. Model output element `k` → cell at `(k ÷ 11, k % 11)` in x-major order, which must be transposed back: `scores[x + 11*y] = avg[x*11 + y]`.

**Rotation invariance**: `encodeBoard` returns both `input1` (original) and `input2` (180° rotation = per-channel element reversal). The worker runs both through the model sequentially (ORT WASM is single-threaded) and `averageOutputs` averages them — `avg[i] = (out1[i] + out2[n-1-i]) / 2` — then applies the blue transpose if needed.

**Border padding**: The 11×11 board is encoded as 13×13 with a 1-cell border. Border cells are not zero — they encode the connectivity of each player's goal edges, which helps the CNN learn win conditions without seeing the literal graph structure.

### Frontend: worker protocol

`ai/modelWorker.ts` is a module Web Worker (bundled separately by Vite via `?worker` URL). The main thread sends a single message type:

```
→ INFER_PAIR  { input1: Float32Array, input2: Float32Array, boardSize: number, modelUrl: string }
← RESULT_PAIR { out1: Float32Array, out2: Float32Array }   // buffers transferred (zero-copy)
← ERROR       { message: string }
```

The model URL must be an absolute URL resolved on the main thread (`document.baseURI`) because the worker's `import.meta.url` resolves relative URLs against the worker bundle in `assets/`, not the page root where the `.onnx` file lives.

Session loading is deduped: `ensureLoaded` uses a module-level promise so concurrent messages during cold-start share one load.

### Frontend: game state machine

`gameReducer` in `game/state.ts` drives all state transitions:

```
idle  ──PLAYER_MOVE──►  thinking  ──AI_MOVE / AI_SURE_WIN──►  idle
                                                             └──► gameover (if winner)
```

`useAI` (`hooks/useAI.ts`) observes `status === "thinking"` and drives the AI turn: it first runs `findSureWinMove` synchronously (minimax depth 1 then 3); if no forced win is found, it encodes the board and posts to the worker. Stale worker responses (e.g. after a game reset) are discarded by comparing a `stateKey` snapshot taken at request time against the current state when the response arrives.

### Frontend: pie rule (swap rule)

After the human's first stone, clicking that same occupied cell triggers `SWAP` — the human effectively takes the agent's perspective (colors flip). When the **AI** decides to swap (it evaluates whether the first stone is too strong), it returns the occupied cell id as its move; the reducer detects `cells[cellId] !== null` with `numMoves === 1` and flips `agentIsBlue` without placing a new stone. `aiSwapped` is set in state so the UI can inform the player.

### Configuration
All training and model hyperparameters live in `conf/` as Hydra/OmegaConf YAML files. Two self-contained presets: `conf/preset/dev.yaml` (3×3, small model, 2 iterations — default) and `conf/preset/prod.yaml` (11×11, 18-layer model, 5000 iterations). Switch with `preset=prod`; override individual values with e.g. `train.learning_rate=3e-4`. Hydra writes a fully resolved config + logs for each run under `outputs/YYYY-MM-DD/HH-MM-SS/.hydra/`.
**Hyperparameter sweeps (TODO):** `bayesian_optimization.py` (scikit-optimize) was removed. The replacement is [Hydra's Optuna sweeper](https://hydra.cc/docs/plugins/optuna_sweeper/) — `hydra-optuna-sweeper` needs to be added to `pyproject.toml` and a sweep config written at `conf/hydra/sweeper/optuna.yaml`.

### Python board representation
Input: 2-channel tensor (red stones, blue stones) with border padding to help convolutions handle edge conditions.

### Solver tables: file format

Solver output (`hexhex/solver/solve.py` → `tables/NxN.bin`) is a single binary file with three sections:

1. **Header (14 B)** — `struct.pack("<4sHQ", b"HXSV", board_size, num_entries)`.
2. **Keys** — `uint64[num_entries]`, **sorted ascending**, written as raw little-endian bytes. Each key is the base-3 encoding of one position: `key = sum_i digit(i) * 3^i` where cell index `i = x * n + y` and digit is 0 (empty), 1 (red), 2 (blue). Fits in u64 for `n ≤ 5`.
3. **Values** — packed bitarray (`numpy.packbits(..., bitorder="little")`) of length `num_entries` bits, padded to a whole byte. Bit `i` is 1 iff side-to-move wins from position `keys[i]`.

Lookup is `np.searchsorted(keys, key)` then a bit check; missing keys (positions never reached, e.g. illegal stone-count parity) raise `KeyError`. Storage cost is dominated by keys (~8.1 B per entry); to scale up further the keys + searchsorted layer can be swapped for a minimal perfect hash without touching the consumer interface.

Pie rule is intentionally **disabled** in the solver: with the pie rule the second player trivially wins on solved boards (they swap whenever the first move is too strong), so the table would only ever record losses for red.
