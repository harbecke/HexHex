#!/usr/bin/env python3
from configparser import ConfigParser

import torch

from hex.creation import create_data, create_model
from hex.elo import elo
from hex.evaluation import win_position
from hex.training import train
from hex.utils.logger import logger
from hex.utils.utils import load_model


class RepeatedSelfTrainer:
    def __init__(self, config_file):
        self.config = ConfigParser()
        self.config.read(config_file)
        self.model_names = []
        self.data_files = []
        self.tournament_results = None

    def repeated_self_training(self):
        model_name = self.create_initial_model()
        for i in range(self.config.getint('REPEATED SELF TRAINING', 'num_iterations', fallback=100)):
            data_file = self.create_data_samples(model_name)
            self.data_files.append(data_file)
            new_model_name = '%s_%04d' % (self.config.get('CREATE MODEL', 'model_name'), i)
            data_file = self.prepare_training_data()
            self.train_model(model_name, new_model_name, data_file)
            model_name = new_model_name
            self.model_names.append(model_name)
            self.create_all_elo_ratings()
            win_position.win_count(f'models/{model_name}.pt', self.config['VS RANDOM'])

    def create_initial_model(self):
        config = self.config['CREATE MODEL']
        self.model_names += [config['model_name']]
        create_model.create_and_store_model(config)
        return config['model_name']

    def create_data_samples(self, model_name):
        model = load_model(f'models/{model_name}.pt')

        self_play_args = self.config['CREATE DATA']
        self_play_args['data_range_min'] = '0'
        self_play_args['data_range_max'] = '1'
        self_play_args['run_name'] = model_name

        create_data.create_self_play_data(
                self_play_args, model
        )
        return self_play_args.get('run_name')

    def prepare_training_data(self):
        """
        gathers the last n training samples and writes them into a separate file.
        :return: filename of file with training data
        """
        n = self.config.getint('REPEATED SELF TRAINING', 'train_samples_pool_size')
        x, y, z = torch.Tensor(), torch.LongTensor(), torch.Tensor()
        for file in self.data_files[::-1]:
            x_new, y_new, z_new = torch.load('data/' + file + '_0.pt')
            x, y, z = torch.cat([x, x_new]), torch.cat([y, y_new]), torch.cat([z, z_new])
            if x.shape[0] >= n:
                break
        if x.shape[0] > n:
            x, y, z = x[:n], y[:n], z[:n]
        torch.save((x, y, z), 'data/current_training_data_0.pt')
        return 'current_training_data'

    def train_model(self, input_model, output_model, data_file):
        config = self.config['TRAIN']
        config['load_model'] = input_model
        config['save_model'] = output_model
        config['data_range_min'] = '0'
        config['data_range_max'] = '1'
        config['validation_split'] = '0'
        config['data'] = data_file
        train.train(config)

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
        model_with_ratings = list(zip(ratings, self.model_names))
        model_with_ratings.sort(reverse=True)

        output = ['{:6} {}'.format(int(rating), model) for rating, model in model_with_ratings]
        for line in output:
            logger.info(line)

        with open('ratings.txt', 'w') as file:
            file.write('   ELO Model\n')
            file.write('\n'.join(output))


if __name__ == '__main__':
    trainer = RepeatedSelfTrainer('config.ini')
    trainer.repeated_self_training()
