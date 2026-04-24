# Train your own agent!

## Play against a pretrained agent

It is easy to play against our agent. A fully trained model is already included in the repository. Python 3.10 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
git clone https://github.com/harbecke/hexhex && cd hexhex
uv sync
uv run python -m hexhex.interactive.interactive
```

Note that both players can make use of the switch rule.
Some starting moves are much better than others, the switch rule forces the first player to pick a more neutral move.

## Setup and Installation

We use [uv](https://github.com/astral-sh/uv) to manage all dependencies and the virtual environment.

1. **Install uv**: See the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).
2. **Sync Environment**:
   ```bash
   uv sync
   ```

## Configuration

All hyperparameters live in `conf/` as typed YAML files (via [Hydra](https://hydra.cc)):

```
conf/
  config.yaml       # shared settings (evaluate, interactive) + defaults to preset: dev
  preset/
    dev.yaml        # 3×3 board, small model, 2 RST iterations — default
    prod.yaml       # 11×11 board, 18-layer model, 5000 iterations — produced models/11_2w4_2000.pt
```

Switch presets or override individual values on the command line:

```bash
# Full production training run
uv run python -m hexhex.training.repeated_self_training preset=prod

# Override a single value without changing preset
uv run python -m hexhex.training.repeated_self_training train.learning_rate=3e-4
```

Hydra writes a resolved copy of the config and logs for each run under `outputs/`:

```
outputs/
  YYYY-MM-DD/
    HH-MM-SS/
      .hydra/
        config.yaml   # fully resolved config for this run
        overrides.yaml
      hexhex.log      # run log (if file logging is configured)
```

## Running Training and Tools

```bash
# Repeated self-training (main training loop)
uv run python -m hexhex.training.repeated_self_training

# Interactive GUI (play against the agent)
uv run python -m hexhex.interactive.interactive interactive.model=my_model_name

# Evaluate two models head-to-head
uv run python -m hexhex.evaluation.evaluate_two_models \
    evaluate.model1=model_a evaluate.model2=model_b

# Create puzzle data
uv run python -m hexhex.creation.puzzle

```

## Reference Models

Training uses `reference_models.json`. Use `"random"` for a random-play reference, or a model name string for a previously trained model.

## Visualize Training

```bash
uv run tensorboard --logdir runs/
```

## Hyperparameter Sweeps (TODO)

The old `bayesian_optimization.py` (scikit-optimize) has been removed. The recommended replacement is [Hydra's Optuna sweeper](https://hydra.cc/docs/plugins/optuna_sweeper/):

```bash
pip install hydra-optuna-sweeper
uv run python -m hexhex.training.repeated_self_training \
    --multirun \
    train.learning_rate='interval(1e-5,1e-2)' \
    model.layers='range(2,12)'
```

This requires adding `hydra-optuna-sweeper` to `pyproject.toml` and a sweep config at `conf/hydra/sweeper/optuna.yaml`. Not yet wired up in this repo.

## Testing

```bash
uv run pytest
```

## Features

* Board representation with logic + switch rule
* CNN to evaluate positions (18 layers, 64 channels, skip connections)
* Batch-wise self-play for dataset generation
* Iterative self-play → train → evaluate loop
* ELO rating via Bradley-Terry model
* Puzzle set for endgame evaluation
* Typed YAML config with Hydra CLI overrides
* Playable GUI (Pygame)
