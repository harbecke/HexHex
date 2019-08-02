#!/usr/bin/env python3
import json
import os
from collections import defaultdict
from configparser import ConfigParser

import torch

from hex.creation import create_data, create_model
from hex.elo import elo
from hex.evaluation import win_position
from hex.model.hexconvolution import RandomModel
from hex.training import train
from hex.utils.logger import logger
from hex.utils.summary import writer
from hex.utils.utils import load_model


def load_reference_models(config):
    reference_model_path = 'reference_models.json'
    if not os.path.isfile(reference_model_path):
        with open(reference_model_path, 'w') as file:
            file.write("{}")
    with open(reference_model_path) as file:
        reference_models = json.load(file)
    board_size_str = str(config['CREATE MODEL'].getint('board_size'))
    if board_size_str not in reference_models:
        reference_models[board_size_str] = []
        with open(reference_model_path, 'w') as file:
            file.write(json.dumps(reference_models, indent=4))
    return reference_models[board_size_str]


class RepeatedSelfTrainer:
    def __init__(self, config_file):
        self.config = ConfigParser()
        self.config.read(config_file)
        self.num_data_models = self.config.getint('REPEATED SELF TRAINING', 'num_data_models')
        self.samples_per_model = self.config.getint('CREATE DATA', 'samples_per_model')
        self.model_name = self.config.get('CREATE MODEL', 'model_name')
        self.model_names = []
        self.start_index = self.config.getint('REPEATED SELF TRAINING', 'start_index', fallback=0)
        self.end_index = self.start_index + self.config.getint('REPEATED SELF TRAINING', 
            'num_iterations', fallback=100)
        self.tournament_results = defaultdict(lambda: defaultdict(int))
        self.reference_models = load_reference_models(self.config)

    def get_model_name(self, i):
        return '%s_%04d' % (self.model_name, i)

    def get_data_files(self, i):
        return [self.get_model_name(idx) for idx in range(i)]

    def repeated_self_training(self):
        self.current_data = list(self.initial_data())

        if self.start_index == 0:
            self.create_initial_model()
            self.model_names = [self.get_model_name(0)]

        while len(self.current_data[0]) < self.num_data_models * self.samples_per_model:
            new_data_triple = self.create_data_samples(self.get_model_name(self.start_index))
            for idx in range(3):
                self.current_data[idx] = torch.cat((self.current_data[idx], new_data_triple[idx]),0)

        for idx in range(3):
            self.current_data[idx] = self.current_data[idx][:self.num_data_models * self.samples_per_model]

        for i in range(self.start_index+1, self.end_index+1):
            start = ((i-1) % self.num_data_models)*self.samples_per_model
            end = start + self.samples_per_model
            new_data_triple = self.create_data_samples(self.get_model_name(i-1))
            for idx in range(3):
                self.current_data[idx][start : end] = new_data_triple[idx]
            self.train_model(self.get_model_name(i-1), self.get_model_name(i), self.current_data)
            self.model_names.append(self.get_model_name(i))
            self.create_all_elo_ratings()
            self.measure_win_counts(self.get_model_name(i))

        torch.save(self.current_data, f'data/{self.model_name}.pt')
        logger.info(f'self-play data generation wrote data/{self.model_name}.pt')

    def create_initial_model(self):
        config = self.config['CREATE MODEL']
        create_model.create_and_store_model(config, self.get_model_name(0))
        return

    def create_data_samples(self, model_name):
        model = load_model(f'models/{model_name}.pt')
        self_play_args = self.config['CREATE DATA']
        self_play_args['samples_per_model'] = str(self.samples_per_model)
        return create_data.create_self_play_data(self_play_args, model)

    def initial_data(self):
        if self.config.getboolean('REPEATED SELF TRAINING', 'load_initial_data'):
            logger.info('=== loading initial data ===')
            logger.info('')
            return torch.load(f'data/{self.model_name}.pt')

        else:
            logger.info('=== writing initial data ===')
            logger.info('')
            model = RandomModel(self.config.getint('CREATE MODEL', 'board_size'))
            self_play_args = self.config['CREATE DATA']
            self_play_args['samples_per_model'] = str(self.num_data_models * self.samples_per_model)
            return create_data.create_self_play_data(self_play_args, model)

    def train_model(self, input_model, output_model, data):
        config = self.config['TRAIN']
        config['load_model'] = input_model
        config['save_model'] = output_model
        train.train(config, data)

    def create_all_elo_ratings(self):
        """
        Incrementally updates ELO ratings by playing all games between latest model and all other models.
        """
        logger.info("")
        logger.info("=== Updating ELO ratings ===")
        if len(self.model_names) <= 1:
            return
        args = self.config['ELO']
        self.tournament_results = elo.add_to_tournament(
            self.model_names[:-1],
            self.model_names[-1],
            args,
            self.tournament_results
        )
        ratings = elo.create_ratings(self.tournament_results)
        writer.add_scalar('elo', ratings[self.model_names[-1]])

        all_model_names = list(ratings.keys())
        all_model_names.sort(key=lambda name: ratings[name], reverse=True)

        output = ['{:6} {}'.format(int(ratings[model]), model) for model in all_model_names]
        for line in output:
            logger.info(line)

        with open('ratings.txt', 'w') as file:
            file.write('   ELO Model\n')
            file.write('\n'.join(output))

    def measure_win_counts(self, model_name):
        reference_models = {
            'random': RandomModel(self.config.getint('CREATE MODEL', 'board_size'))
        }
        for model in self.reference_models:
            reference_models[model] = load_model(f'models/{model}.pt')
        win_position.win_count(f'models/{model_name}.pt', reference_models, self.config['VS REFERENCE MODELS'])


if __name__ == '__main__':
    trainer = RepeatedSelfTrainer('config.ini')
    trainer.repeated_self_training()
