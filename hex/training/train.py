#!/usr/bin/env python3
import argparse
import os
import time
from configparser import ConfigParser

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler

from hex.utils.logger import logger
from hex.utils.utils import device, load_model, create_optimizer, load_optimizer


def get_args(config_file):
    config = ConfigParser()
    config.read(config_file)
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', type=str, default=config.get('TRAIN', 'load_model'))
    parser.add_argument('--save_model', type=str, default=config.get('TRAIN', 'save_model'))
    parser.add_argument('--data', type=str, default=config.get('TRAIN', 'data'))
    parser.add_argument('--data_range_min', type=int, default=config.getint('TRAIN', 'data_range_min'))
    parser.add_argument('--data_range_max', type=int, default=config.getint('TRAIN', 'data_range_max'))
    parser.add_argument('--weight_decay', type=float, default=config.getfloat('TRAIN', 'weight_decay'))
    parser.add_argument('--batch_size', type=int, default=config.getint('TRAIN', 'batch_size'))
    parser.add_argument('--epochs', type=float, default=config.getfloat('TRAIN', 'epochs'))
    parser.add_argument('--optimizer', type=str, default=config.get('TRAIN', 'optimizer'))
    parser.add_argument('--optimizer_load', type=bool, default=config.getboolean('TRAIN', 'optimizer_load'))
    parser.add_argument('--learning_rate', type=float, default=config.getfloat('TRAIN', 'learning_rate'))
    parser.add_argument('--validation_bool', type=bool, default=config.getboolean('TRAIN', 'validation_bool'))
    parser.add_argument('--validation_data', type=str, default=config.get('TRAIN', 'validation_data'))
    parser.add_argument('--validation_split', type=float, default=config.getfloat('TRAIN', 'validation_split'))
    parser.add_argument('--print_loss_frequency', type=int, default=config.getint('TRAIN', 'print_loss_frequency'))
    return parser.parse_args(args=[])


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
        param_loss = self.args.weight_decay * \
                     sum(torch.pow(p, 2).sum() for p in self.model.parameters())
        return param_loss, policy_loss, value_loss


def train_model(model, save_model_path, dataloader, criterion, optimizer, epochs, device, weight_decay,
                print_loss_frequency, validation_triple):
    log_file = os.path.join('logs', save_model_path + '.csv')

    with open(log_file, 'w') as log:
        log.write('# batch val_loss pred_loss weighted_param_loss duration[s]\n')

    start = time.time()

    '''
    trains model with backpropagation, loss criterion is currently binary cross-entropy and optimizer is adadelta
    '''
    for epoch in range(epochs):

        running_loss = 0.0

        for i, (board_states, moves, labels) in enumerate(dataloader, 0):
            board_states, moves, labels = board_states.to(device), moves.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = torch.exp(model(board_states))
            output_values = torch.gather(outputs, 1, moves)

            loss = criterion(output_values.view(-1), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % print_loss_frequency == 0:
                l2loss = sum(torch.pow(p, 2).sum() for p in model.parameters() if p.requires_grad)
                weighted_param_loss = weight_decay * l2loss

                if validation_triple is not None:
                    with torch.no_grad():
                        val_pred_tensor = torch.exp(model(validation_triple[0]))
                        val_values = torch.gather(val_pred_tensor, 1, validation_triple[1])
                        val_loss = criterion(val_values.view(-1), validation_triple[2])

                    duration = int(time.time() - start)
                    logger.info('batch %3d / %3d val_loss: %.3f  pred_loss: %.3f  l2_param_loss: %.3f weighted_param_loss: %.3f'
                            % (i + 1, len(dataloader), val_loss, loss.item(), l2loss, weighted_param_loss))

                    with open(log_file, 'a') as log:
                        log.write(f'{i + 1} {val_loss} {loss.item()} {weighted_param_loss} {duration}\n')

                else:
                    logger.info('batch %3d / %3d pred_loss: %.3f  l2_param_loss: %.3f weighted_param_loss: %.3f'
                          % (i + 1, len(dataloader), loss.item(), l2loss, weighted_param_loss))

        l2loss = sum(torch.pow(p, 2).sum() for p in model.parameters() if p.requires_grad)
        weighted_param_loss = weight_decay * l2loss
        logger.info('Epoch [%d] pred_loss: %.3f l2_param_loss: %.3f weighted_param_loss: %.3f'
              % (epoch + 1, running_loss/(i+1), l2loss, weighted_param_loss))
    
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

    if config.getint('epochs') < 1:
        concat_len = train_dataset.__len__()
        sampler = SubsetRandomSampler(torch.randperm(concat_len)[:int(concat_len * config.getfloat('epochs'))])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.getint('batch_size'), sampler=sampler,
                                                     num_workers=0)
        config['epochs'] = '1'

    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.getint('batch_size'), shuffle=True,
                                                     num_workers=0)

    model_file = f'models/{config.get("load_model")}.pt'
    model = load_model(model_file)
    nn.DataParallel(model).to(device)

    optimizer_weight_decay = config.getfloat('weight_decay')

    optimizer = create_optimizer(optimizer_type=config.get('optimizer'), parameters=model.parameters(),
        optimizer_weight_decay=optimizer_weight_decay, learning_rate=config.getfloat('learning_rate'))

    if config.getboolean('optimizer_load'):
        optimizer = load_optimizer(optimizer, model_file)

    val_triple = None
    if config.getboolean('validation_bool'):
        val_board_tensor, val_moves_tensor, val_target_tensor = torch.load(f'data/{config.get("validation_data")}.pt')
        val_triple = (val_board_tensor.to(device), val_moves_tensor.to(device), val_target_tensor.to(device))

    criterion = lambda pred, y: 0.8*nn.L1Loss(reduction='mean')(pred, y)+0.2*nn.BCELoss(reduction='mean')(pred, y)
    trained_model, trained_optimizer = train_model(model, config.get('save_model'), train_loader, criterion, optimizer,
        int(config.getfloat('epochs')), device, config.getfloat('weight_decay'), config.getint('print_loss_frequency'), val_triple)

    checkpoint = torch.load(model_file, map_location=device)
    checkpoint['model_state_dict'] = trained_model.state_dict()
    file_name = f'models/{config.get("save_model")}.pt'
    torch.save(checkpoint, file_name)
    logger.info(f'wrote {file_name}')
