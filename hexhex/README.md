# Train your own agent!
Sorry, this is poorly documented and likely outdated. You're mostly on your own here.

## Play against a pretrained agent

It is easy to play against our agent. A fully trained model is already included in the repository. Python 3.10 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
git clone https://github.com/harbecke/hexhex && cd hexhex
# Install dependencies and sync environment
uv sync
# Run the agent
uv run python -m hexhex
```

Note that both players can make use of the switch rule.
Some starting moves are much better than others, the switch rule forces the first player to pick a more neutral move.
This means after the first player (red) makes the first move, the second player (blue) can switch colors and take over this first move.
The game continues by the first player (now blue) making the second move.

## Setup and Installation

We use [uv](https://github.com/astral-sh/uv) to manage all dependencies and the virtual environment. This ensures a consistent and reproducible setup.

1.  **Install uv**: Please refer to the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/) for instructions on how to install `uv` on your system.
2.  **Sync Environment**: Once `uv` is installed, run the following command in the project root to create a virtual environment and install all necessary dependencies:
    ```bash
    uv sync
    ```

## Execution

All scripts and notebooks must be run through `uv run` from the project root.

1.  **Initialize Configuration**: Copy the sample configuration files to the root directory:
    ```bash
    cp sample_files/* .
    ```

2.  **Run Training and Tools**:
    - **Repeated Self-Training**: `uv run python -m hexhex.training.repeated_self_training`
    - **Bayesian Optimization**: `uv run python -m hexhex.training.bayesian_optimization`
    - **Interactive GUI**: `uv run python -m hexhex.interactive.interactive`

3.  **Reference Models**: Both training scripts use `reference_models.json`.
    - Use `"random"` for a random reference model for your board size.
    - Use `"{your_model_name}"` to use a previously trained model as a reference.

### Visualize Training with Tensorboard
Training automatically creates log files for Tensorboard. View them using:
```bash
uv run python -m tensorboard.main --logdir runs/
```

### Visualize Bayesian optimization with jupyter
Bayesian optimization creates a pickle file in `data/bayes_experiments`.
Run jupyter notebook, select the correct experiment in the second cell, and execute:
```bash
uv run jupyter notebook
```

### Loading Into hexgui
[hexgui](https://github.com/ryanbhayward/hexgui) can be used for interactive play and replaying `FF4` files.
To load the AI into hexgui:
1.  Go to `Program -> new program`.
2.  Enter the command to start `play_cli.py`. Since `uv` is required, use a bash script:
    ```bash
    #!/bin/bash
    cd /path/to/hex
    uv run python -m hexhex.interactive.play_cli
    ```
3.  Connect via `Program -> connect local program`.

### Testing
We use [pytest](https://docs.pytest.org/) for unit and integration testing. Run the test suite using:
```bash
uv run pytest
```

## Features

* board representation with logic + switch rule
* network to evaluate positions
  * output activation of network is sigmoid for each stone
  * these are probabilities of how likely that stone wins the game
  * loss function is between prediction of selected stone and outcome of game
* creating models with hyperparameters
* batch-wise self-play to generate datasets
* training and validating models
* evaluating models against each other
* ELO rating via `output_ratings` in `hexhex/elo/elo.py`
* iterative training loop
* puzzle set for endgame evaluation
* config to control plenty of hyperparameters
* Bayesian optimization to tune hyperparameters
* playable gui
