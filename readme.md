# HexHex v0.6

AlphaGo Zero adaptation of Hex. [Image of intend](https://user-images.githubusercontent.com/33026629/32346749-47b65b36-c049-11e7-9bac-08bc42cf9dae.png)

See [here](https://www.gwern.net/docs/rl/2017-silver.pdf) for full paper.


## Getting Started

### Manual Installation

* Python >= 3.6
* Pytorch (see [here](https://pytorch.org/get-started/locally/) for installation info)

* install pygame `pip install pygame` for interactive

### Installation with pipenv

```
# install pipenv to manage dependencies
pip install pipenv 

# install dependencies
# use --skip-lock due to a recent regression in pipenv: https://github.com/pypa/pipenv/issues/2284
pipenv install --skip-lock

# activate virtual environment
pipenv shell 

# create model, training data, train model, and evaluate
./run_example.py
```

### Execution
* test installation with `run_example.py`

* copy sample_config as initial configuration file `cp sample_config.ini config.ini`
* change parameters in config.ini
* run scripts
    - `./create_model.py`
    - `./create_data.py`
    - `./train.py`
    - `./evaluate_two_models.py`
    - `./interactive.py`
    - `./repeated_data_and_training.py`

## Features

* board representation with logic + switch rule

* network to evaluate positions
  * output activation of network is sigmoid for each stone
  * these are probabilities of how likely that stone wins the game
  * loss function is bewteen prediction of selected stone and outcome of game

* scripts for
  * creating models with hyperparameters
  * batch-wise self-play to generate datasets
  * training and validating models
  * evaluating models against each other
  * iterative training loop

* config to control plenty of hyperparameters

* playable gui `interactive.py`

* trained model `five_board_wd0.001.pt` for 5x5 board

* puzzle validation set `puzzle.pt` for endgame strategy evaluation

## To-dos

* implement Monte Carlo tree search

* everything...
