# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HexHex is a reinforcement learning agent for the board game Hex, trained via self-play without MCTS (unlike AlphaGo Zero). The project has two components:
- **Python backend** (`hexhex/`): RL training pipeline, game logic, neural network
- **React frontend** (`app/`): Browser-based UI using ONNX.js for in-browser inference

Live demo: https://cleeff.github.io/hex/

## Commands

### Python (Backend)
```bash
uv sync                                                     # Install dependencies
uv run pytest                                               # Run all tests
uv run pytest tests/test_file.py::test_name                # Run a single test
uv run python -m hexhex.training.repeated_self_training    # Run training loop
uv run python -m hexhex.interactive.interactive             # Launch interactive GUI
uv run tensorboard.main --logdir runs/                      # Visualize training
```

### JavaScript (Frontend — app/)
```bash
npm start        # Dev server
npm run build    # Production build
npm test         # Run tests
```

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
- `create_model.py`: instantiates models from `config.ini` hyperparameters
- `create_data.py`: generates batches of self-play games
- `noise.py`: exploration noise for move selection

**`hexhex/training/`** — Training pipeline
- `train.py`: core training loop
- `repeated_self_training.py`: iterative self-play → train → evaluate loop
- `bayesian_optimization.py`: hyperparameter search via scikit-optimize

**`hexhex/evaluation/` + `hexhex/elo/`** — Model ranking
- `evaluate_two_models.py`: head-to-head match
- `elo.py`: Bradley-Terry model for multi-agent ranking

**`hexhex/export/onnx_export.py`** — Converts trained PyTorch model to ONNX for web deployment

**`app/src/App.js`** — React frontend; loads ONNX model in-browser via onnxjs, renders 11×11 board with react-hexgrid, uses boardgame.io for state management

### Configuration
All training and model hyperparameters live in `config.ini`. Template with defaults is in `sample_files/config.ini`. Key sections: `[CREATE MODEL]`, `[CREATE DATA]`, `[TRAIN]`, `[REPEATED SELF TRAINING]`, `[ELO]`.

### Board representation
Input: 2-channel tensor (red stones, blue stones) with border padding to help convolutions handle edge conditions.
