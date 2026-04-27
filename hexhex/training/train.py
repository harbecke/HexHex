#!/usr/bin/env python3
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset

from hexhex.creation import puzzle
from hexhex.utils.logger import logger
from hexhex.utils.paths import run_model_path
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


def train_model(model, train_dataloader, val_dataloader, optimizer, puzzle_triple, cfg, global_step_offset=0):
    criterion = lambda pred, y: 0.8*nn.L1Loss(reduction='sum')(pred, y)+0.2*nn.BCELoss(reduction='sum')(pred, y)

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

    weight_decay = cfg.weight_decay
    epochs = math.ceil(cfg.epochs)
    for epoch in range(epochs):
        train_loss_avg = Average()
        step = global_step_offset + epoch

        for i, train_triple in enumerate(train_dataloader):
            optimizer.zero_grad()

            train_loss = measure_loss(train_triple, eval_mode=False)
            train_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
            optimizer.step()

            train_loss_avg.add(train_loss.item(), len(train_triple[0]))

            if i % cfg.print_loss_frequency == 0:
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
                writer.add_scalar('train/grad_norm', grad_norm, step)
                writer.add_scalar('train/puzzle_loss', puzzle_loss.mean(), step)
                writer.add_scalar('train/l2_weights', l2loss, step)

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
        writer.add_scalar('train/train_loss', train_loss_avg.mean(), step)
        writer.add_scalar('train/val_loss', val_loss.mean(), step)

    logger.info('=== finished model training ===')
    return model, optimizer


def train(cfg, training_data, validation_data, load_model_name, save_model_name, puzzle_num_samples=1000, global_step_offset=0):
    """
    loads data and sets criterion and optimizer for train_model
    """
    logger.info("")
    logger.info("=== training model ===")

    train_dataset = TensorDataset(training_data[0], training_data[1], training_data[2])
    val_dataset = TensorDataset(validation_data[0], validation_data[1], validation_data[2])

    if cfg.epochs < 1:
        total_train_sample = len(train_dataset)
        num_train_samples = int(total_train_sample * cfg.epochs)
        train_dataset, _ = torch.utils.data.random_split(train_dataset,
                                                         [num_train_samples, total_train_sample - num_train_samples])

    batch_size = cfg.batch_size
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    model_file = run_model_path(load_model_name)
    model = load_model(model_file)
    nn.DataParallel(model).to(device)

    optimizer = create_optimizer(
        optimizer_type=cfg.optimizer,
        parameters=model.parameters(),
        learning_rate=cfg.learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay
    )

    puzzle_file = f'data/{model.board_size}_puzzle.pt'
    if not os.path.exists(puzzle_file):
        logger.info("")
        logger.info("=== creating missing puzzle file ===")
        puzzle.create_puzzle(model.board_size, puzzle_num_samples)

    puzzle_triple = torch.load(puzzle_file, map_location=device)

    trained_model, trained_optimizer = train_model(model=model,
                                                   train_dataloader=train_loader,
                                                   val_dataloader=val_loader,
                                                   optimizer=optimizer,
                                                   puzzle_triple=puzzle_triple,
                                                   cfg=cfg,
                                                   global_step_offset=global_step_offset)

    checkpoint = torch.load(model_file, map_location=device)
    checkpoint['model_state_dict'] = trained_model.state_dict()
    file_name = run_model_path(save_model_name)
    torch.save(checkpoint, file_name)
    logger.info(f'wrote {file_name}')
