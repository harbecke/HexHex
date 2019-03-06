#!/usr/bin/env python3
import torch
from configparser import ConfigParser

import train
from hex.creation import create_data, create_model
from hex.evaluation import evaluate_two_models


def repeated_self_training(config_file):
    """
    Runs a self training loop.
    Each iteration produces a new model which is then trained on self-play data
    """
    config = ConfigParser()
    config.read(config_file)

    prefix = config.get('REPEATED SELF TRAINING', 'prefix')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    board_size = config.getint('CREATE DATA', 'board_size')

    create_model.create_model_from_config_file(config_file)

    initial_model = 'first_random'
    champion_iter = 1
    champion_filename = initial_model

    for model_id in range(10000):
        champion = torch.load(f'models/{champion_filename}.pt', map_location=device)

        create_data.generate_data_files(
                file_counter_start=config.getint('CREATE DATA', 'data_range_min'),
                file_counter_end=config.getint('CREATE DATA', 'data_range_max'),
                samples_per_file=config.getint('CREATE DATA', 'samples_per_file'),
                model=champion,
                device=device,
                run_name=champion_filename,
                noise=config.get('CREATE DATA', 'noise'),
                noise_parameters=[float(parameter) for parameter in config.get('CREATE DATA', 'noise_parameters').split(",")],
                temperature=config.getfloat('CREATE DATA', 'temperature'),
                board_size=config.getint('CREATE DATA', 'board_size'),
                batch_size=config.getint('CREATE DATA', 'batch_size'),
                temperature_decay=1
        )

        train_args = train.get_args(config_file)
        train_args.load_model = f'{prefix}_{model_id - 1}' if model_id > 0 else initial_model
        train_args.save_model = f'{prefix}_{model_id}'
        train_args.data = champion_filename
        train.train(train_args)

        eval_args = evaluate_two_models.get_args(config_file)
        result, signed_chi_squared = evaluate_two_models.play_games(
                models=[torch.load(f'models/{prefix}_{model_id}.pt'), champion],
                openings=eval_args.openings,
                number_of_games=eval_args.number_of_games,
                device=device,
                batch_size=eval_args.batch_size,
                board_size=eval_args.board_size,
                temperature=eval_args.temperature,
                temperature_decay=eval_args.temperature_decay,
                plot_board=eval_args.plot_board
        )
        win_rate = (result[0][0] + result[1][0]) / (sum(result[0]) + sum(result[1]))
        print(f'win_rate = {win_rate}')
        if win_rate  > .55:
            champion_filename = f'{prefix}_{model_id}'
            champion_iter = 1
            print(f'Accept {champion_filename} as new champion!')
        else:
            champion_iter += 1
            print(f'The champion remains in place. Iteration: {champion_iter}')


if __name__ == '__main__':
    repeated_self_training(config_file='repeated_self_training.ini')
