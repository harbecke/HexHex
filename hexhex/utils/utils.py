#!/usr/bin/env python3
import random as _random

import numpy as np
import torch
import torch.optim as optim

from hexhex.creation.create_model import create_model
from hexhex.utils.logger import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    """Seed all RNGs for reproducible runs.

    Covers every randomness source in the training pipeline: Python `random`
    (opening shuffle in evaluate_two_models), NumPy (data shuffle in
    create_data), and torch (move sampling in temperature/hexgame, RandomModel
    init data, model weight init, DataLoader shuffle, random_split). Also
    forces cuDNN into deterministic mode and disables the kernel autotuner.

    Pass `None` to skip seeding entirely (keeps cuDNN benchmark on, faster
    on GPU but produces different results across runs).
    """
    if seed is None:
        return
    seed = int(seed)
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"seeded all RNGs with seed={seed} (cuDNN: deterministic=True, benchmark=False)")


def _one_pass(iters):
    for it in iters:
        try:
            yield next(it)
        except StopIteration:
            pass


def zip_list_of_lists(*iterables):
    iters = [iter(it) for it in iterables]
    output_list = []
    while True:
        iter_list = list(_one_pass(iters))
        output_list.extend(list(iter_list))
        if iter_list==[]:
            return output_list


def correct_position1d(position1d, board_size, player):
    if player:
        return position1d//board_size + (position1d%board_size)*board_size
    else:
        return position1d


def load_model(model_file, export_mode=False):
    checkpoint = torch.load(model_file, map_location=device)
    model = create_model(checkpoint['config'], export_mode)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    torch.no_grad()
    return model


def create_optimizer(optimizer_type, parameters, learning_rate, momentum, weight_decay):
    logger.debug("=== creating optimizer ===")
    if optimizer_type == 'adadelta':
        return optim.Adadelta(parameters, weight_decay=weight_decay)
    elif optimizer_type == 'rmsprop':
        return optim.RMSprop(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        return optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        return optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        logger.error(f'Unknown optimizer {optimizer_type}')
        raise SystemExit


def get_targets(boards, gamma):
    target_list = [[0.5 + 0.5 * (-1) ** k * (1 - gamma) ** (2 * (k//2)) for k in reversed(range(len(
        board.move_history)))] for board in boards]
    return torch.tensor(zip_list_of_lists(*target_list), device=torch.device('cpu'))


class Average:
    def __init__(self):
        self.num_samples = 0
        self.total = 0.0

    def add(self, value, num_samples):
        self.num_samples += num_samples
        self.total += value

    def mean(self):
        try:
            return self.total / self.num_samples
        except ZeroDivisionError:
            return float("NaN")
