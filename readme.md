# HexHex v0.3

AlphaGo Zero adaptation of Hex. Image of intend: [Image of intend:](https://user-images.githubusercontent.com/33026629/32346749-47b65b36-c049-11e7-9bac-08bc42cf9dae.png)

See [here](https://www.gwern.net/docs/rl/2017-silver.pdf) for full paper.


## Getting Started

Install prerequisites and run "play.ipynb" notebook for an iteration of self-play.

### Prerequisites

* Python 3

* Pytorch (see [here](https://pytorch.org/get-started/locally/) for installation info)

* Jupyter Notebook

* for GUI pygame

* for images pillow


## Features

* board representation with logic + switch rule

* network to evaluate positions
  * output activation of network is sigmoid for each stone
  * these are probabilities of how likely that stone wins the game
  * loss function is cross-entropy of prediction of selected stone and outcome of game

* plenty of hyperparameters

* little documentation

* self-play dataset script

* train models script


## To-dos (somewhat chronological)

* more documentation

* evaluate net against older version with ELO

* playability

* implement Monte Carlo tree search