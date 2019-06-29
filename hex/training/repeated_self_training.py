#!/usr/bin/env python3
from configparser import ConfigParser

import torch

from hex.creation import create_data, create_model
from hex.elo import elo
from hex.training import train
from hex.utils.utils import dotdict
from hex.utils.utils import load_model


class RepeatedSelfTrainer:
    def __init__(self, config_file):
        self.config = ConfigParser()
        self.config.read(config_file)
        self.model_names = []
        self.data_files = []

    def repeated_self_training(self):
        model_name = self.create_initial_model()
        for i in range(100):
            data_file = self.create_data_samples(model_name)
            self.data_files.append(data_file)
            new_model_name = '%s_%04d' % (self.config.get('REPEATED SELF TRAINING', 'name'), i)
            data_file = self.prepare_training_data()
            self.train_model(model_name, new_model_name, data_file)
            model_name = new_model_name
            self.create_elo_ratings(model_name)
            self.model_names.append(model_name)
            self.create_all_elo_ratings()

    def create_initial_model(self):
        model_creation_args = dotdict({
            'model_type': self.config.get('REPEATED SELF TRAINING', 'model_type'),
            'board_size': self.config.getint('REPEATED SELF TRAINING', 'board_size'),
            'layers': self.config.getint('REPEATED SELF TRAINING', 'layers'),
            'intermediate_channels': self.config.getint('REPEATED SELF TRAINING', 'intermediate_channels'),
            'layer_type': self.config.get('REPEATED SELF TRAINING', 'layer_type'),
            'model_name': self.config.get('REPEATED SELF TRAINING', 'name') + '_initial'
        })
        create_model.create_model_from_args(model_creation_args)
        return model_creation_args.model_name

    def create_data_samples(self, model_name):
        model, _ = load_model(f'models/{model_name}.pt')

        self_play_args = self.config['CREATE DATA']
        self_play_args['data_range_min'] = '0'
        self_play_args['data_range_max'] = '1'
        self_play_args['samples_per_file'] = self.config.get('REPEATED SELF TRAINING', 'samples_per_model')
        self_play_args['run_name'] = model_name
        self_play_args['noise_epsilon'] = self.config.get('REPEATED SELF TRAINING', 'noise_epsilon')
        self_play_args['noise_spread'] = self.config.get('REPEATED SELF TRAINING', 'noise_spread')
        self_play_args['temperature'] = self.config.get('REPEATED SELF TRAINING', 'temperature')

        create_data.create_self_play_data(
                self_play_args, model
        )
        return self_play_args.get('run_name')

    def prepare_training_data(self):
        N = self.config.getint('REPEATED SELF TRAINING', 'train_samples_pool_size')
        x, y, z = torch.Tensor(), torch.LongTensor(), torch.Tensor()
        for file in self.data_files[::-1]:
            x_new,y_new,z_new = torch.load('data/' + file + '_0.pt')
            x, y, z = torch.cat([x, x_new]), torch.cat([y, y_new]), torch.cat([z, z_new])
            if x.shape[0] >= N:
                break
        if x.shape[0] > N:
            x, y, z = x[:N], y[:N], z[:N]
        torch.save((x, y, z), 'data/current_training_data_0.pt')
        return 'current_training_data'

    def train_model(self, input_model, output_model, data_file):
        training_args = dotdict({
            'load_model': input_model,
            'save_model': output_model,
            'data': data_file,
            'data_range_min': 0,
            'data_range_max': 1,
            'batch_size': self.config.getint('TRAIN', 'batch_size'),
            'optimizer': self.config.get('TRAIN', 'optimizer'),
            'optimizer_load': self.config.getboolean('TRAIN', 'optimizer_load'),
            'learning_rate': self.config.getfloat('TRAIN', 'learning_rate'),
            'validation_bool': False,
            'epochs': self.config.getfloat('TRAIN', 'epochs'),
            'samples_per_epoch': self.config.getint('TRAIN', 'samples_per_epoch'),
            'weight_decay': self.config.getfloat('TRAIN', 'weight_decay'),
            'validation_split': 0.,
            'print_loss_frequency': self.config.getint('TRAIN', 'print_loss_frequency')
        })
        train.train(training_args)

    def create_all_elo_ratings(self):
        if len(self.model_names) <= 1:
            return
        args = self.config['ELO']
        elo.output_ratings(self.model_names, args=args)

    def create_elo_ratings(self, latest_model):
        # TODO would like to incrementally update elo ratings here
        pass


if __name__ == '__main__':
    trainer = RepeatedSelfTrainer('config.ini')
    trainer.repeated_self_training()
