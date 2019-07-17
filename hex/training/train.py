#!/usr/bin/env python3
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler

from hex.utils.logger import logger
from hex.utils.summary import writer
from hex.utils.utils import device, load_model, create_optimizer, load_optimizer


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


class Training:
    def __init__(self, args, model, optimizer):
        self.args = args
        self.model = model
        self.log_file = os.path.join('logs', args.save_model + '.csv')
        with open(self.log_file, 'w') as log:
            log.write('# batch val_loss pred_loss weighted_param_loss duration[s]\n')
        self.device = device
        self.optimizer = optimizer
        self.stats = TrainingStats()

    def train_mcts_model(self, train_dataset, val_dataset):
        mean_epoch_loss_history = {'train': [], 'val': []}
        for epoch in range(int(self.args.epochs)):
            sampler = SubsetRandomSampler(torch.randperm(len(train_dataset))[:self.args.samples_per_epoch])
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, sampler=sampler)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.batch_size)
            dataloaders = {'train': train_loader, 'val': val_loader}

            for phase in ['val', 'train']: # run validation first, s.t. loss values are from the same network
                for i, (board_states, policies_train, value_train) in enumerate(dataloaders[phase], 0):
                    board_states, policies_train, value_train \
                        = board_states.to(self.device), policies_train.to(self.device), value_train.to(self.device)

                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        param_loss, policy_loss, value_loss = self.measure_mean_losses(
                                board_states,
                                policies_train,
                                value_train
                        )

                        loss = policy_loss + value_loss + param_loss

                        loss_triple = LossTriple(value_loss.item(), policy_loss.item(), param_loss.item())
                        self.stats.add_batch(phase, epoch, i, loss_triple)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                # mean_epoch_loss_history[phase].append(
                #         {key: value / len(dataloaders[phase].dataset) for key, value in running_losses.items()}
                # )

            logger.info('Epoch [%3d] mini-batches [%8d] %s'
                        % (epoch, epoch * len(train_loader), self.stats.last_values_to_string())
                        )

        logger.debug('Finished Training\n')
        return self.model, self.optimizer

    def measure_mean_losses(self, board_states, policies_train, value_train):
        policies_log, values_out = self.model(board_states)
        policies_train_entropy = torch.distributions.Categorical(probs=policies_train).entropy()
        # this calculated cross entropy as policy output is log(p)
        # subtracting policies_train_entropy, s.t. policy_loss == 0 if both policies are equal
        policy_loss = torch.mean(torch.sum(-policies_log * policies_train, dim=1) - policies_train_entropy)
        value_loss_fct = nn.MSELoss(reduction='mean')
        value_loss = value_loss_fct(values_out, value_train)
        param_loss = self.args.weight_decay * sum(torch.pow(p, 2).sum() for p in self.model.parameters())
        return param_loss, policy_loss, value_loss


def train_model(model, dataloader, optimizer, puzzle_triple, config):
    log_file = os.path.join('logs', config.get('save_model') + '.csv')

    with open(log_file, 'w') as log:
        log.write('# batch val_loss pred_loss weighted_param_loss duration[s]\n')

    start = time.time()

    criterion = lambda pred, y: 0.8*nn.L1Loss(reduction='mean')(pred, y)+0.2*nn.BCELoss(reduction='mean')(pred, y)
    weight_decay = config.getfloat('weight_decay')
    epochs = math.ceil(config.getfloat('epochs'))
    for epoch in range(epochs):

        running_loss = 0.0

        for i, (board_states, moves, labels) in enumerate(dataloader, 0):
            board_states, moves, labels = board_states.to(device), moves.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = torch.sigmoid(model(board_states))
            output_values = torch.gather(outputs, 1, moves)

            loss = criterion(output_values.view(-1), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print_loss_frequency = config.getint('print_loss_frequency')
            if i % print_loss_frequency == 0:
                l2loss = sum(torch.pow(p, 2).sum() for p in model.parameters() if p.requires_grad)
                weighted_param_loss = weight_decay * l2loss

                if puzzle_triple is not None:
                    with torch.no_grad():
                        puzzle_pred_tensor = torch.sigmoid(model(puzzle_triple[0]))
                        puzzle_values = torch.gather(puzzle_pred_tensor, 1, puzzle_triple[1])
                        puzzle_loss = criterion(puzzle_values.view(-1), puzzle_triple[2])

                    duration = int(time.time() - start)
                    logger.info(
                        'batch %3d / %3d '
                        'puzzle_loss: %.3f '
                        'pred_loss: %.3f '
                        'l2_param_loss: %.3f '
                        'weighted_param_loss: %.3f'
                        % (i + 1, len(dataloader), puzzle_loss, loss.item(), l2loss, weighted_param_loss)
                    )
                    writer.add_scalar('train/val_loss', puzzle_loss)
                    writer.add_scalar('train/l2_weights', l2loss)

                    with open(log_file, 'a') as log:
                        log.write(f'{i + 1} {puzzle_loss} {loss.item()} {weighted_param_loss} {duration}\n')

                else:
                    # TODO: remove this case and generate puzzle data if non-existent
                    logger.info('batch %3d / %3d pred_loss: %.3f  l2_param_loss: %.3f weighted_param_loss: %.3f'
                          % (i + 1, len(dataloader), loss.item(), l2loss, weighted_param_loss))

        l2loss = sum(torch.pow(p, 2).sum() for p in model.parameters() if p.requires_grad)
        weighted_param_loss = weight_decay * l2loss
        pred_loss = running_loss / (i + 1)
        logger.info('Epoch [%d] pred_loss: %.3f l2_param_loss: %.3f weighted_param_loss: %.3f'
              % (epoch + 1, pred_loss, l2loss, weighted_param_loss))
        writer.add_scalar('train/pred_loss', pred_loss)

    logger.debug('Finished Training\n')
    return model, optimizer


def train(config):
    """
    loads data and sets criterion and optimizer for train_model
    """
    logger.info("")
    logger.info("=== training model ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_list = []

    for idx in range(config.getint('data_range_min'), config.getint('data_range_max')):
        board_states, moves, targets = torch.load('data/{}_{}.pt'.format(config.get('data'), idx))
        dataset_list.append(TensorDataset(board_states, moves, targets))

    concat_dataset = ConcatDataset(dataset_list)
    total_len = len(concat_dataset)
    val_part = int(config.getfloat('validation_split') * total_len)
    train_dataset, val_dataset = torch.utils.data.random_split(concat_dataset, [total_len - val_part, val_part])

    if config.getfloat('epochs') < 1:
        concat_len = train_dataset.__len__()
        sampler = SubsetRandomSampler(torch.randperm(concat_len)[:int(concat_len * config.getfloat('epochs'))])
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config.getint('batch_size'),
                                                   sampler=sampler,
                                                   num_workers=0)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config.getint('batch_size'),
                                                   shuffle=True,
                                                   num_workers=0)

    model_file = f'models/{config.get("load_model")}.pt'
    model = load_model(model_file)
    nn.DataParallel(model).to(device)

    optimizer_weight_decay = config.getfloat('weight_decay')

    optimizer = create_optimizer(
        optimizer_type=config.get('optimizer'),
        parameters=model.parameters(),
        optimizer_weight_decay=optimizer_weight_decay,
        learning_rate=config.getfloat('learning_rate')
    )

    if config.getboolean('optimizer_load'):
        optimizer = load_optimizer(optimizer, model_file)

    puzzle_triple = None
    if config.getboolean('use_puzzle', True):
        puzzle_triple = torch.load(f'data/{model.board_size}_puzzle.pt', map_location=device)

    trained_model, trained_optimizer = train_model(model=model,
                                                   dataloader=train_loader,
                                                   optimizer=optimizer,
                                                   puzzle_triple=puzzle_triple,
                                                   config=config)

    checkpoint = torch.load(model_file, map_location=device)
    checkpoint['model_state_dict'] = trained_model.state_dict()
    file_name = f'models/{config.get("save_model")}.pt'
    torch.save(checkpoint, file_name)
    logger.info(f'wrote {file_name}')
