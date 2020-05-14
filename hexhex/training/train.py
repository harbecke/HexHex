#!/usr/bin/env python3
import copy
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset

from hexhex.creation import puzzle
from hexhex.utils.logger import logger
from hexhex.utils.summary import writer
from hexhex.utils.utils import device, load_model, create_optimizer, Average


class LossTriple:
    def __init__(self, value_loss, policy_loss, param_loss):
        self.value = value_loss
        self.policy = policy_loss
        self.param = param_loss

    def total(self):
        return self.value + self.policy + self.param


class TrainingStats:
    def __init__(self):
        self.stats = {
            'value': {'train': [], 'val': []},
            'policy': {'train': [], 'val': []},
            'param': {'train': [], 'val': []},
        }

    def add_batch(self, phase: str, epoch: int, batch_idx: int, loss: LossTriple):
        for type in self.all_loss_types():
            epochs = self.stats[type][phase]
            if epoch >= len(epochs):
                epochs.append([])
            batches = epochs[epoch]
            if batch_idx >= len(batches):
                batches.append([])
            batch = batches[batch_idx]
            batch.append(loss.__getattribute__(type))

    @staticmethod
    def all_loss_types():
        return ['value', 'policy', 'param']

    def get_epoch_mean(self, loss_type: str, phase: str, epoch = -1):
        if loss_type == 'total':
            return sum(self.get_epoch_mean(t, phase, epoch) for t in self.all_loss_types())
        return np.mean(self.stats[loss_type][phase][epoch]).item()

    def last_values_to_string(self):
        return "total: {:.3f} {:.3f} | value: {:.3f} {:.3f} | policy: {:.3f} {:.3f} | param: {:.3f} {:.3f} | train val".format(
                self.get_epoch_mean('total', 'train'),
                self.get_epoch_mean('total', 'val'),
                self.get_epoch_mean('value', 'train'),
                self.get_epoch_mean('value', 'val'),
                self.get_epoch_mean('policy', 'train'),
                self.get_epoch_mean('policy', 'val'),
                self.get_epoch_mean('param', 'train'),
                self.get_epoch_mean('param', 'val'),
        )


def train_model(model, train_dataloader, val_dataloader, optimizer, puzzle_triple, config):
    criterion = nn.MSELoss(reduction='sum')(pred, y)

    def measure_loss(data_triple, eval_mode):
        def _measure_loss_impl(data_triple):
            board_states, moves, labels = data_triple
            board_states, moves, labels = board_states.to(device), moves.to(device), labels.to(device)
            outputs = torch.sigmoid(model(board_states))
            output_values = torch.gather(outputs, 1, moves)
            return criterion(output_values.view(-1), labels)

        if eval_mode:
            model.eval()
            with torch.no_grad():
                return _measure_loss_impl(data_triple)
        else:
            torch.enable_grad()
            model.train()
            return _measure_loss_impl(data_triple)

    def measure_weight_loss():
        return sum(torch.pow(p, 2).sum() for p in model.parameters() if p.requires_grad)

    weight_decay = config.getfloat('weight_decay')
    epochs = math.ceil(config.getfloat('epochs'))
    for epoch in range(epochs):
        train_loss_avg = Average()

        for i, train_triple in enumerate(train_dataloader):
            optimizer.zero_grad()

            train_loss = measure_loss(train_triple, eval_mode=False)
            train_loss.backward()
            optimizer.step()

            train_loss_avg.add(train_loss.item(), len(train_triple[0]))

            print_loss_frequency = config.getint('print_loss_frequency')
            if i % print_loss_frequency == 0:
                l2loss = measure_weight_loss()
                weighted_param_loss = weight_decay * l2loss

                puzzle_loss = Average()
                if puzzle_triple is not None:
                    puzzle_loss.add(measure_loss(puzzle_triple, eval_mode=True), len(puzzle_triple[0]))

                logger.info(
                    f'batch {i + 1:3} / {len(train_dataloader):3} '
                    f'puzzle_loss: {puzzle_loss.mean():.3f} '
                    f'l2_param_loss: {l2loss:.3f} '
                    f'weighted_param_loss: {weighted_param_loss:.3f}'
                )
                writer.add_scalar('train/puzzle_loss', puzzle_loss.mean())
                writer.add_scalar('train/l2_weights', l2loss)

        val_loss = Average()
        for val_triple in val_dataloader:
            val_loss.add(measure_loss(val_triple, eval_mode=True), len(val_triple[0]))

        l2loss = measure_weight_loss()
        weighted_param_loss = weight_decay * l2loss
        logger.info(
            f'Epoch {epoch + 1} '
            f'train_loss: {train_loss_avg.mean():.3f} '
            f'val_loss: {val_loss.mean():.3f} '
            f'l2_param_loss: {l2loss:.3f} '
            f'weighted_param_loss: {weighted_param_loss:.3f}'
        )
        writer.add_scalar('train/train_loss', train_loss_avg.mean())
        writer.add_scalar('train/val_loss', val_loss.mean())

    writer.close()
    logger.debug('=== finished training ===\n')
    return model, optimizer


def train(config, training_data, validation_data):
    """
    loads data and sets criterion and optimizer for train_model
    """
    logger.info("")
    logger.info("=== training model ===")

    train_dataset = TensorDataset(training_data[0], training_data[1], training_data[2])
    val_dataset = TensorDataset(validation_data[0], validation_data[1], validation_data[2])

    if config.getfloat('epochs') < 1:
        total_train_sample = len(train_dataset)
        num_train_samples = int(total_train_sample * config.getfloat('epochs'))
        train_dataset, _ = torch.utils.data.random_split(train_dataset,
                                                         [num_train_samples, total_train_sample - num_train_samples])

    batch_size = config.getint('batch_size')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    model_file = f'models/{config.get("load_model")}.pt'
    model = load_model(model_file)
    nn.DataParallel(model).to(device)

    optimizer = create_optimizer(
        optimizer_type=config.get('optimizer'),
        parameters=model.parameters(),
        learning_rate=config.getfloat('learning_rate'),
        momentum=config.getfloat('momentum'),
        weight_decay=config.getfloat('weight_decay')
    )

    puzzle_file = f'data/{model.board_size}_puzzle.pt'
    if not os.path.exists(puzzle_file):
        logger.info("")
        logger.info("=== creating missing puzzle file ===")
        puzzle_config = copy.deepcopy(config)
        puzzle_config['board_size'] = str(model.board_size)
        puzzle.create_puzzle(puzzle_config)

    puzzle_triple = torch.load(puzzle_file, map_location=device)

    trained_model, trained_optimizer = train_model(model=model,
                                                   train_dataloader=train_loader,
                                                   val_dataloader=val_loader,
                                                   optimizer=optimizer,
                                                   puzzle_triple=puzzle_triple,
                                                   config=config)

    checkpoint = torch.load(model_file, map_location=device)
    checkpoint['model_state_dict'] = trained_model.state_dict()
    file_name = f'models/{config.get("save_model")}.pt'
    torch.save(checkpoint, file_name)
    logger.info(f'wrote {file_name}')
