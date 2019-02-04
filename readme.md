# HexHex v0.1

AlphaGo Zero adaptation of Hex. Image of intend: ![Image of intend:](https://user-images.githubusercontent.com/33026629/32346749-47b65b36-c049-11e7-9bac-08bc42cf9dae.png)

See [here](https://www.gwern.net/docs/rl/2017-silver.pdf) for full paper.


## Getting Started

Install prerequisites and run "play.ipynb" notebook for an iteration of self-play.

### Prerequisites

* Python 3

* Pytorch (see [here](https://pytorch.org/get-started/locally/) for installation info)

* Juypter Notebook 


## Features

* board representation with logic

* (untrained) neural network to evaluate positions

* creation of datapoint with self-play

* plenty of hyperparameters

* little documentation


## To-dos (somewhat chronological)

* create full pytorch dataset skript

* implement dataloader and backpropagation

* more documentation

* evaluate net against older version with ELO

* playability

* implement Monte Carlo tree search