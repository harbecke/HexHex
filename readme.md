# HexHex v0.2

AlphaGo Zero adaptation of Hex. Image of intend: ![Image of intend:](https://user-images.githubusercontent.com/33026629/32346749-47b65b36-c049-11e7-9bac-08bc42cf9dae.png)

See [here](https://www.gwern.net/docs/rl/2017-silver.pdf) for full paper.


## Getting Started

Install prerequisites and run "play.ipynb" notebook for an iteration of self-play.

### Prerequisites

* Python 3

* Pytorch (see [here](https://pytorch.org/get-started/locally/) for installation info)

* Jupyter Notebook 


## Features

* board representation with logic + switch rule

* (untrained) neural network to evaluate positions

* creation of datapoint with self-play

* plenty of hyperparameters

* little documentation

* create full pytorch dataset skript


## To-dos (somewhat chronological)

* implement dataloader and backpropagation

* more documentation

* evaluate net against older version with ELO

* playability

* implement Monte Carlo tree search