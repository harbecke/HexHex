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
from hex.utils.utils import load_model, merge_dicts_of_dicts


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
        self.train_samples = self.config.getint('CREATE DATA', 'num_train_samples')
        self.val_samples = self.config.getint('CREATE DATA', 'num_val_samples')
        self.model_name = self.config.get('CREATE MODEL', 'model_name')
        self.model_names = []
        self.start_index = self.config.getint('REPEATED SELF TRAINING', 'start_index', fallback=0)
        self.tournament_results = defaultdict(lambda: defaultdict(int))
        self.reference_models = load_reference_models(self.config)

    def get_model_name(self, i):
        return '%s_%04d' % (self.model_name, i)

    def get_data_files(self, i):
        return [self.get_model_name(idx) for idx in range(i)]

    def prepare_rst(self):
        training_data, validation_data = self.initial_data()

        if self.start_index == 0:
            self.create_initial_model()

        self.model_names.append(self.get_model_name(self.start_index))
        self.sorted_model_names = self.model_names[:]

        self.training_data = self.check_enough_data(training_data, self.train_samples)
        self.validation_data = self.check_enough_data(validation_data, self.val_samples)

    def rst_loop(self, i):
        train_samples_per_model = self.train_samples // self.num_data_models
        val_samples_per_model = self.val_samples // self.num_data_models
        start = ((i-1) % self.num_data_models)
        new_train_triple = self.create_data_samples(self.get_model_name(i-1),
            train_samples_per_model)
        new_val_triple = self.create_data_samples(self.get_model_name(i-1),
            val_samples_per_model, verbose=False)
        for idx in range(3):
            self.training_data[idx][start*train_samples_per_model : (start+1) * \
                train_samples_per_model] = new_train_triple[idx]
            self.validation_data[idx][start*val_samples_per_model : (start+1) * \
                val_samples_per_model] = new_val_triple[idx]
        self.train_model(self.get_model_name(i-1), self.get_model_name(i), self.training_data,
            self.validation_data)
        self.model_names.append(self.get_model_name(i))
        self.create_all_elo_ratings()
        self.measure_win_counts(self.get_model_name(i), self.reference_models, verbose=True)

    def repeated_self_training(self):
        self.prepare_rst()

        for i in range(self.start_index + 1, self.start_index + 1 + self.config.
            getint('REPEATED SELF TRAINING', 'num_iterations')):
            self.rst_loop(i)

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
            logger.info("")
            logger.info('=== loading initial data ===')
            return torch.load(f'data/{self.model_name}.pt')

        else:
            logger.info("")
            logger.info('=== creating random initial data ===')
            model = RandomModel(self.config.getint('CREATE MODEL', 'board_size'))
            self_play_args = self.config['CREATE DATA']
            training_data = create_data.create_self_play_data(self_play_args, model,
                self.train_samples, verbose=False)
            validation_data = create_data.create_self_play_data(self_play_args, model,
                self.val_samples, verbose=False)
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
        logger.info("=== updating ELO ratings ===")

        args = self.config['ELO']
        self.tournament_results = elo.add_to_tournament(
            self.sorted_model_names,
            self.model_names[-1],
            args,
            self.tournament_results
        )
        ratings = elo.create_ratings(self.tournament_results)
        writer.add_scalar('elo', ratings[self.model_names[-1]])

        self.sorted_model_names.append(self.model_names[-1])
        self.sorted_model_names.sort(key=lambda name: ratings[name], reverse=True)

        output = ['{:6} {}'.format(int(ratings[model]), model) for model in self.sorted_model_names]
        for line in output:
            logger.info(line)

        with open('ratings.txt', 'w') as file:
            file.write('   ELO Model\n')
            file.write('\n'.join(output))

    def get_best_rating(self):
        for reference_idx in range(1, len(self.reference_models)):
            self.measure_win_counts(self.reference_models[reference_idx],
                self.reference_models[:reference_idx], verbose=False)
        ratings = elo.create_ratings(self.tournament_results)
        best_trained_model = max(self.model_names[1:], key=lambda name: ratings[name])
        best_reference_model = max(self.reference_models + self.model_names[0:1],
            key=lambda name: ratings[name])
        diff = ratings[best_trained_model] - ratings[best_reference_model]
        logger.info(f"ELO difference between best trained model and best reference model: {diff:0.2f}")
        return diff

    def measure_win_counts(self, model_name, reference_model_names, verbose):
        reference_models = {}
        for model in reference_model_names:
            if model == "random":
                reference_models["random"] = RandomModel(self.config.getint('CREATE MODEL', 'board_size'))
            else:
                reference_models[model] = load_model(f'models/{model}.pt')
        results = win_position.win_count(model_name, reference_models,
            self.config['VS REFERENCE MODELS'], verbose)
        self.tournament_results = merge_dicts_of_dicts(self.tournament_results, results)


if __name__ == '__main__':
    config = ConfigParser()
    config.read('config.ini')
    trainer = RepeatedSelfTrainer(config)
    trainer.repeated_self_training()
