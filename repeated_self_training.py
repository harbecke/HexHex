#!/usr/bin/env python3
import torch
from configparser import ConfigParser

import create_data
import create_model
import evaluate_two_models
import train


def repeated_self_training(config_file):
    """
    Runs a self training loop.
    Each iteration produces a new model which is then trained on self-play data
    """
    config = ConfigParser()
    config.read(config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    create_model.create_model_from_config_file(config_file)

    initial_model = '5_random'
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
                noise_alpha=config.getfloat('CREATE DATA', 'noise_alpha'),
                temperature=config.getfloat('CREATE DATA', 'temperature'),
                board_size=config.getint('CREATE DATA', 'board_size'),
                batch_size=config.getint('CREATE DATA', 'batch_size'),
        )

        train_args = train.get_args(config_file)
        train_args.load_model = f'5_gen{model_id - 1}' if model_id > 0 else initial_model
        train_args.save_model = f'5_gen{model_id}'
        train_args.data = champion_filename
        train.train(train_args)

        result, signed_chi_squared = evaluate_two_models.play_games(
                models=[torch.load(f'models/5_gen{model_id}.pt'), champion],
                number_of_games=config.getint('EVALUATE MODELS', 'number_of_games'),
                device=device,
                temperature=config.getfloat('EVALUATE MODELS', 'temperature'),
                board_size=config.getint('EVALUATE MODELS', 'board_size'),
                plot_board=config.getboolean('EVALUATE MODELS', 'plot_board'),
                batch_size=config.getint('EVALUATE MODELS', 'batch_size')
        )
        if (result[0][0] + result[1][0]) / (sum(result[0]) + sum(result[1]))  > .55:
            champion_filename = f'5_gen{model_id}'
            champion_iter = 1
            print(f'Accept {champion_filename} as new champion!')
        else:
            champion_iter += 1
            print(f'The champion remains in place. Iteration: {champion_iter}')


if __name__ == '__main__':
    repeated_self_training(config_file='repeated_self_training.ini')
