#!/usr/bin/env python3
import math
from collections import defaultdict

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from hexhex.creation import create_data, create_model
from hexhex.elo import elo
from hexhex.evaluation import win_position
from hexhex.model.hexconvolution import RandomModel
from hexhex.training import train
from hexhex.utils.logger import logger
from hexhex.utils.summary import writer
from hexhex.utils.utils import load_model, merge_dicts_of_dicts


class RepeatedSelfTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_data_models = cfg.rst.num_data_models
        self.train_samples = cfg.data.num_train_samples
        self.val_samples = cfg.data.num_val_samples
        self.model_name = cfg.model.model_name
        self.model_names = []
        self.start_index = cfg.rst.start_index
        self.tournament_results = defaultdict(lambda: defaultdict(int))
        self.reference_models = list(cfg.vs_reference.reference_models)
        self.total_epochs_trained = 0

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
        self.measure_win_counts(self.get_model_name(i), self.reference_models, verbose=True)

    def repeated_self_training(self):
        self.prepare_rst()

        for i in range(self.start_index + 1, self.start_index + 1 + self.cfg.rst.num_iterations):
            self.rst_loop(i)

        if self.cfg.rst.save_data:
            torch.save((self.training_data, self.validation_data), f'data/{self.model_name}.pt')
            logger.info(f'self-play data generation wrote data/{self.model_name}.pt')

        logger.info('=== finished training ===')

    def create_initial_model(self):
        create_model.create_and_store_model(self.cfg.model, self.get_model_name(0))

    def create_data_samples(self, model_name, num_samples, verbose=True):
        model = load_model(f'models/{model_name}.pt')
        return create_data.create_self_play_data(self.cfg.data, model, num_samples, verbose)

    def initial_data(self):
        if self.cfg.rst.load_initial_data:
            logger.info("")
            logger.info('=== loading initial data ===')
            return torch.load(f'data/{self.model_name}.pt')

        else:
            logger.info("")
            logger.info('=== creating random initial data ===')
            model = RandomModel(self.cfg.model.board_size)
            training_data = create_data.create_self_play_data(self.cfg.data, model,
                self.train_samples, verbose=False)
            validation_data = create_data.create_self_play_data(self.cfg.data, model,
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
        train.train(
            cfg=self.cfg.train,
            training_data=training_data,
            validation_data=validation_data,
            load_model_name=input_model,
            save_model_name=output_model,
            puzzle_num_samples=self.cfg.puzzle.num_samples,
            global_step_offset=self.total_epochs_trained,
        )
        self.total_epochs_trained += math.ceil(self.cfg.train.epochs)

    def create_all_elo_ratings(self):
        """
        Incrementally updates ELO ratings by playing all games between latest model and all other models.
        """
        logger.info("")
        logger.info("=== updating ELO ratings ===")

        self.tournament_results = elo.add_to_tournament(
            self.sorted_model_names,
            self.model_names[-1],
            self.cfg.elo,
            self.tournament_results
        )
        ratings = elo.create_ratings(self.tournament_results)
        writer.add_scalar('elo', ratings[self.model_names[-1]], len(self.model_names) - 1)

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
                reference_models["random"] = RandomModel(self.cfg.model.board_size)
            else:
                reference_models[model] = load_model(f'models/{model}.pt')
        results = win_position.win_count(model_name, reference_models,
            self.cfg.vs_reference, verbose)
        self.tournament_results = merge_dicts_of_dicts(self.tournament_results, results)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir
    preset = HydraConfig.get().runtime.choices.get("preset", "unknown")
    g = "\033[32m"
    r = "\033[0m"
    print(f"{g}{'='*50}{r}")
    print(f"{g}  HexHex self-play training{r}")
    print(f"{g}{'='*50}{r}")
    print(f"{g}  This run is logged to:{r}")
    print(f"{g}    {output_dir}/{r}")
    print(f"{g}{r}")
    print(f"{g}  That directory contains:{r}")
    print(f"{g}    repeated_self_training.log  — full console output{r}")
    print(f"{g}    .hydra/config.yaml          — fully resolved config (preset: {preset}){r}")
    print(f"{g}    .hydra/overrides.yaml       — any CLI overrides applied{r}")
    print(f"{g}{'='*50}{r}")
    trainer = RepeatedSelfTrainer(cfg)
    trainer.repeated_self_training()


if __name__ == '__main__':
    main()
