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

    champion_iter = 1
    champion_filename = '5_random'

    for model_id in range(100):
        champion = torch.load(f'models/{champion_filename}.pt', map_location=device)

        create_data.generate_data_files(
                number_of_files=config.getint('CREATE DATA', 'number_of_files'),
                samples_per_file=config.getint('CREATE DATA', 'samples_per_file'),
                model=champion,
                device=device,
                run_name=champion_filename,
                noise_level=config.getfloat('CREATE DATA', 'noise_level'),
                noise_alpha=config.getfloat('CREATE DATA', 'noise_alpha'),
                temperature=config.getfloat('CREATE DATA', 'temperature'),
                board_size=config.getint('CREATE DATA', 'board_size')
        )

        train_args = train.get_args(config_file)
        train_args.load_model = champion_filename if champion_iter == 1 else f'5_gen{model_id}'
        train_args.save_model = f'5_gen{model_id}'
        train_args.data = champion_filename
        train.train(train_args)

        result = evaluate_two_models.play_games(
                model1=torch.load(f'models/5_gen{model_id}.pt'),
                model2=champion,
                number_of_games=config.getint('EVALUATE MODELS', 'number_of_games'),
                device=device,
                temperature=config.getfloat('EVALUATE MODELS', 'temperature'),
                board_size=config.getint('EVALUATE MODELS', 'board_size'),
                plot_board=config.getboolean('EVALUATE MODELS', 'plot_board')
        )
        if result[0] / sum(result) > .55:
            champion_filename = f'5_gen{model_id}'
            champion_iter = 1
            print(f'Accept {champion_filename} as new champion!')
        else:
            champion_iter += 1
            print(f'The champion remains in place. Iteration: {champion_iter}')


if __name__ == '__main__':
    repeated_self_training(config_file='repeated_self_training.ini')
