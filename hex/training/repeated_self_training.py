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
    def __init__(self, config):
        self.config = config
        self.num_data_models = self.config.getint('REPEATED SELF TRAINING', 'num_data_models')
        self.train_samples = self.config.getint('CREATE DATA', 'train_samples_per_model')
        self.val_samples = self.config.getint('CREATE DATA', 'val_samples_per_model')
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
        training_data, validation_data = self.initial_data()

        if self.start_index == 0:
            self.create_initial_model()

        self.model_names.append(self.get_model_name(self.start_index))
        self.sorted_model_names = self.model_names[:]

        training_data = self.check_enough_data(training_data, self.train_samples * self.num_data_models)
        validation_data = self.check_enough_data(validation_data, self.val_samples * self.num_data_models)

        for i in range(self.start_index+1, self.end_index+1):
            start = ((i-1) % self.num_data_models)
            new_train_triple = self.create_data_samples(self.get_model_name(i-1), self.train_samples)
            new_val_triple = self.create_data_samples(self.get_model_name(i-1), self.val_samples, verbose=False)
            for idx in range(3):
                training_data[idx][start*self.train_samples : (start+1)* self.train_samples] = new_train_triple[idx]
                validation_data[idx][start*self.val_samples : (start+1)* self.val_samples] = new_val_triple[idx]
            self.train_model(self.get_model_name(i-1), self.get_model_name(i), training_data, validation_data)
            self.model_names.append(self.get_model_name(i))
            self.create_all_elo_ratings()
            self.measure_win_counts(self.get_model_name(i))

        if self.config.getboolean('REPEATED SELF TRAINING', 'save_data'):
            torch.save((training_data, validation_data), f'data/{self.model_name}.pt')
            logger.info(f'self-play data generation wrote data/{self.model_name}.pt')

        logger.info('=== finished training ===')

    def create_initial_model(self):
        config = self.config['CREATE MODEL']
        create_model.create_and_store_model(config, self.get_model_name(0))
        return

    def create_data_samples(self, model_name, num_samples, verbose=True):
        model = load_model(f'models/{model_name}.pt')
        self_play_args = self.config['CREATE DATA']
        return create_data.create_self_play_data(self_play_args, model, num_samples, verbose)

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
            training_data = create_data.create_self_play_data(self_play_args, model,
                self.num_data_models * self.train_samples, verbose=False)
            validation_data = create_data.create_self_play_data(self_play_args, model,
                self.num_data_models * self.val_samples, verbose=False)
            return training_data, validation_data

    def check_enough_data(self, data, amount):
        if len(data[0]) < amount:
            new_data_triple = self.create_data_samples(self.get_model_name(self.start_index),
                amount - len(data[0]), verbose=False)
            for idx in range(3):
                data[idx] = torch.cat((data[idx], new_data_triple[idx]), 0)
            return data
        else:
            return [data_part[:amount] for data_part in data]

    def train_model(self, input_model, output_model, training_data, validation_data):
        config = self.config['TRAIN']
        config['load_model'] = input_model
        config['save_model'] = output_model
        train.train(config, training_data, validation_data)

    def create_all_elo_ratings(self):
        """
        Incrementally updates ELO ratings by playing all games between latest model and all other models.
        """
        logger.info("")
        logger.info("=== Updating ELO ratings ===")

        args = self.config['ELO']
        self.tournament_results = elo.add_to_tournament(
            self.sorted_model_names,
            self.model_names[-1],
            args,
            self.tournament_results
        )
        self.ratings = elo.create_ratings(self.tournament_results)
        writer.add_scalar('elo', self.ratings[self.model_names[-1]])

        self.sorted_model_names.append(self.model_names[-1])
        self.sorted_model_names.sort(key=lambda name: self.ratings[name], reverse=True)

        output = ['{:6} {}'.format(int(self.ratings[model]), model) for model in self.sorted_model_names]
        for line in output:
            logger.info(line)

        with open('ratings.txt', 'w') as file:
            file.write('   ELO Model\n')
            file.write('\n'.join(output))

    def get_best_rating(self):
        return int(self.ratings[self.sorted_model_names[0]]) if int(self.ratings[self.
            sorted_model_names[0]]) != 0 else int(self.ratings[self.sorted_model_names[1]])

    def measure_win_counts(self, model_name):
        reference_models = {
            'random': RandomModel(self.config.getint('CREATE MODEL', 'board_size'))
        }
        for model in self.reference_models:
            reference_models[model] = load_model(f'models/{model}.pt')
        win_position.win_count(f'models/{model_name}.pt', reference_models, self.config['VS REFERENCE MODELS'])


if __name__ == '__main__':
    config = ConfigParser()
    config.read('config.ini')
    trainer = RepeatedSelfTrainer(config)
    trainer.repeated_self_training()
