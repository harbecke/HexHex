#!/usr/bin/env python3
import math
import os
import shutil
import time
from collections import defaultdict

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from hexhex.creation import create_data, create_model
from hexhex.elo import elo
from hexhex.evaluation import win_position
from hexhex.model.hexconvolution import RandomModel
from hexhex.solver.metrics import OptimalityChecker
from hexhex.solver.table import SolutionTable
from hexhex.training import train
from hexhex.utils.logger import logger
from hexhex.utils.paths import (
    RUNS_DIR,
    auto_model_name,
    reference_model_path,
    run_data_path,
    run_model_path,
    run_models_dir,
    saved_data_path,
    set_run_dir,
)
from hexhex.utils.summary import writer
from hexhex.utils.utils import load_model, merge_dicts_of_dicts


class RepeatedSelfTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_data_models = cfg.rst.num_data_models
        self.train_samples = cfg.data.num_train_samples
        self.val_samples = cfg.data.num_val_samples
        self.model_name = auto_model_name(cfg.model)
        self.model_names = []
        self.start_index = cfg.rst.start_index
        self.tournament_results = defaultdict(lambda: defaultdict(int))
        self.reference_models = list(cfg.vs_reference.reference_models)
        self.total_epochs_trained = 0
        self.ratings_file = 'ratings.txt'
        self.optimality_checker = self._load_optimality_checker()

    def _load_optimality_checker(self) -> OptimalityChecker | None:
        n = self.cfg.model.board_size
        path = os.path.join("tables", f"{n}x{n}.bin")
        if not os.path.exists(path):
            logger.warning(
                f"no solver table at {path}; ground-truth optimality metrics will be skipped. "
                f"Generate it with: uv run python -m hexhex.solver.solve --size {n} --out {path}"
            )
            return None
        table = SolutionTable(path)
        logger.info(
            f"loaded solution table {path} ({table.num_entries:,} positions); "
            "will log data/optimality_rate each iteration"
        )
        return OptimalityChecker(table)

    def get_model_name(self, i):
        return '%s_%04d' % (self.model_name, i)

    def get_data_files(self, i):
        return [self.get_model_name(idx) for idx in range(i)]

    def prepare_rst(self):
        training_data, validation_data = self.initial_data()

        if self.start_index == 0:
            self.create_initial_model()
        else:
            self._resume_from_prior_run()

        self.model_names.append(self.get_model_name(self.start_index))
        self.sorted_model_names = self.model_names[:]

        self.training_data = self.check_enough_data(training_data, self.train_samples)
        self.validation_data = self.check_enough_data(validation_data, self.val_samples)

    def rst_loop(self, i):
        t_rst = time.time()
        train_samples_per_model = self.train_samples // self.num_data_models
        val_samples_per_model = self.val_samples // self.num_data_models
        start = ((i-1) % self.num_data_models)

        t_data = time.time()
        new_train_triple = self.create_data_samples(self.get_model_name(i-1),
            train_samples_per_model, step=i, measure_optimality=True)
        new_val_triple = self.create_data_samples(self.get_model_name(i-1),
            val_samples_per_model, verbose=False, step=i)
        writer.add_scalar('time/data_generation', time.time() - t_data, i)

        for idx in range(3):
            self.training_data[idx][start*train_samples_per_model : (start+1) * \
                train_samples_per_model] = new_train_triple[idx]
            self.validation_data[idx][start*val_samples_per_model : (start+1) * \
                val_samples_per_model] = new_val_triple[idx]

        t_train = time.time()
        self.train_model(self.get_model_name(i-1), self.get_model_name(i), self.training_data,
            self.validation_data)
        writer.add_scalar('time/training', time.time() - t_train, i)

        self.model_names.append(self.get_model_name(i))

        t_eval = time.time()
        self.measure_win_counts(self.get_model_name(i), self.reference_models, verbose=True, step=i)
        writer.add_scalar('time/evaluation', time.time() - t_eval, i)

        t_elo = time.time()
        self.create_all_elo_ratings()
        writer.add_scalar('time/elo_tournament', time.time() - t_elo, i)

        writer.add_scalar('time/rst_iteration', time.time() - t_rst, i)

    def repeated_self_training(self):
        self.prepare_rst()

        for i in range(self.start_index + 1, self.start_index + 1 + self.cfg.rst.num_iterations):
            self.rst_loop(i)

        if self.cfg.rst.save_data:
            data_file = run_data_path()
            torch.save((self.training_data, self.validation_data), data_file)
            logger.info(f'self-play data generation wrote {data_file}')

        logger.info('=== finished repeated self-training ===')

    def create_initial_model(self):
        create_model.create_and_store_model(self.cfg.model, self.get_model_name(0))

    def _resume_from_prior_run(self):
        resume_from = self.cfg.rst.resume_from
        if not resume_from:
            raise ValueError(
                f"rst.start_index={self.start_index} requires rst.resume_from=<prior_exp_id> "
                "so the checkpoint can be copied into this run's models/ dir"
            )
        src_path = os.path.join(RUNS_DIR, resume_from, "models",
                                f"{self.get_model_name(self.start_index)}.pt")
        if not os.path.exists(src_path):
            raise FileNotFoundError(
                f"resume checkpoint not found: {src_path}. "
                f"Verify rst.resume_from='{resume_from}' and rst.start_index={self.start_index}."
            )
        dst_path = run_model_path(self.get_model_name(self.start_index))
        shutil.copy2(src_path, dst_path)
        logger.info(f"resumed from {src_path} -> {dst_path}")

    def create_data_samples(self, model_name, num_samples, verbose=True, step=None,
                            measure_optimality=False):
        model = load_model(run_model_path(model_name))
        return create_data.create_self_play_data(
            self.cfg.data, model, num_samples, verbose, step=step,
            optimality_checker=self.optimality_checker if measure_optimality else None,
        )

    def initial_data(self):
        if self.cfg.rst.load_initial_data:
            data_file = saved_data_path(self.cfg.rst.load_initial_data)
            logger.info("")
            logger.info(f'=== loading initial data from {data_file} ===')
            return torch.load(data_file)

        else:
            logger.info("")
            logger.info(f'=== creating random initial data ({self.train_samples} train, {self.val_samples} val samples from random self-play) ===')
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
        new_model = self.model_names[-1]
        opponents = self.sorted_model_names[:self.cfg.elo.max_num_opponents]
        logger.info("")
        logger.info("=== updating ELO ratings ===")

        self.tournament_results = elo.add_to_tournament(
            self.sorted_model_names,
            new_model,
            self.cfg.elo,
            self.tournament_results
        )
        ratings = elo.create_ratings(self.tournament_results)
        writer.add_scalar('elo', ratings[new_model], len(self.model_names) - 1)

        for opponent in opponents:
            new_wins = self.tournament_results[new_model][opponent]
            opp_wins = self.tournament_results[opponent][new_model]
            logger.info(f'  {new_model} vs {opponent}: {new_wins} - {opp_wins}')

        self.sorted_model_names.append(self.model_names[-1])
        self.sorted_model_names.sort(key=lambda name: ratings[name], reverse=True)

        top5 = self.sorted_model_names[:5]
        total = len(self.sorted_model_names)
        for model in top5:
            logger.info('  {:6} {}'.format(int(ratings[model]), model))
        if total > 5:
            logger.info(f'  ... ({total - 5} more not shown)')

        with open(self.ratings_file, 'w') as file:
            file.write('   ELO Model\n')
            file.write('\n'.join('{:6} {}'.format(int(ratings[m]), m) for m in self.sorted_model_names))

        table = ['| ELO | Model |', '|--:|---|']
        for model in self.sorted_model_names:
            table.append(f'| {int(ratings[model])} | {model} |')
        writer.add_text('elo_table', '\n'.join(table), len(self.model_names) - 1)

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

    def measure_win_counts(self, model_name, reference_model_names, verbose, step=None):
        reference_models = {}
        for model in reference_model_names:
            if model == "random":
                reference_models["random"] = RandomModel(self.cfg.model.board_size)
            else:
                reference_models[model] = load_model(reference_model_path(model))
        results = win_position.win_count(model_name, reference_models,
            self.cfg.vs_reference, verbose, step=step)
        self.tournament_results = merge_dicts_of_dicts(self.tournament_results, results)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    run_dir = HydraConfig.get().runtime.output_dir  # runs/<exp_id>
    exp_id = os.path.basename(run_dir)

    set_run_dir(run_dir)
    writer.init(log_dir=run_dir)
    writer.add_text("config", f"```yaml\n{OmegaConf.to_yaml(cfg, resolve=True)}\n```")
    writer.add_text("exp_id", exp_id)

    g = "\033[32m"
    r = "\033[0m"
    print(f"{g}{'='*60}{r}")
    print(f"{g}  HexHex self-play training{r}")
    print(f"{g}{'='*60}{r}")
    print(f"{g}  Experiment id: {exp_id}{r}")
    print(f"{g}{r}")
    print(f"{g}  All artifacts under: {run_dir}/{r}")
    print(f"{g}    repeated_self_training.log     — full console output{r}")
    print(f"{g}    ratings.txt                    — final ELO ranking{r}")
    print(f"{g}    models/{auto_model_name(cfg.model)}_NNNN.pt          — per-iteration checkpoints{r}")
    print(f"{g}    events.out.tfevents.*          — TensorBoard scalars/text{r}")
    print(f"{g}    .hydra/config.yaml             — fully resolved config{r}")
    print(f"{g}    .hydra/overrides.yaml          — CLI overrides applied{r}")
    print(f"{g}{r}")
    print(f"{g}  To visualize training:{r}")
    print(f"{g}    uv run tensorboard --logdir runs/{r}")
    print(f"{g}{'='*60}{r}")

    trainer = RepeatedSelfTrainer(cfg)
    trainer.ratings_file = os.path.join(run_dir, "ratings.txt")
    trainer.repeated_self_training()


if __name__ == '__main__':
    main()
