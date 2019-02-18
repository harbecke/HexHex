#!/usr/bin/env python3
import torch
from configparser import ConfigParser

import create_data
import evaluate_two_models
import train


def repeated_self_training(config_file, data_step, runs, win_rate):
    """
    Runs a self training loop.
    Each iteration produces a new model which is then trained on self-play data
    """
    config = ConfigParser()
    config.read(config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    champion_iter = 1
    champion_filename = config.get('CREATE DATA', 'model')
    data_range_min=config.getint('CREATE DATA', 'data_range_min')
    data_range_max=config.getint('CREATE DATA', 'data_range_max')
    data_range=[data_range_min, data_range_min+data_step]
    for run in range(runs):
        champion = torch.load(f'models/{champion_filename}.pt', map_location=device)

        create_data.generate_data_files(
                file_counter_start=data_range[0],
                file_counter_end=data_range[1],
                samples_per_file=config.getint('CREATE DATA', 'samples_per_file'),
                batch_size = config.getint('CREATE DATA', 'batch_size'),
                model=champion,
                device=device,
                run_name=config.get('CREATE DATA', 'run_name'),
                noise_level=config.getfloat('CREATE DATA', 'noise_level'),
                noise_alpha=config.getfloat('CREATE DATA', 'noise_alpha'),
                temperature=config.getfloat('CREATE DATA', 'temperature'),
                board_size=config.getint('CREATE DATA', 'board_size')
        )
        new_data_range_max = data_range[1]+data_step
        if new_data_range_max > data_range_max:
            data_range=[data_range_min, data_range_min+data_step]
        else:
            data_range=[data_range[0]+data_step, data_range[1]+data_step]

        train_args = train.get_args(config_file)
        train_args.load_model = champion_filename
        train_args.save_model = f'5_gen{champion_iter}'
        train.train(train_args)

        result = evaluate_two_models.play_games(
                models=(torch.load(f'models/5_gen{champion_iter}.pt'), champion),
                number_of_games=config.getint('EVALUATE MODELS', 'number_of_games'),
                batch_size=config.getint('EVALUATE MODELS', 'batch_size'),
                device=device,
                temperature=config.getfloat('EVALUATE MODELS', 'temperature'),
                board_size=config.getint('EVALUATE MODELS', 'board_size'),
                plot_board=config.getboolean('EVALUATE MODELS', 'plot_board')
        )
        if result[0] / sum(result) > win_rate:
            champion_filename = f'5_gen{champion_iter}'
            champion_iter += 1
            print(f'Accept {champion_filename} as new champion!')
        else:
            print(f'The champion remains in place. Iteration: {run}')

def _main():
    config_file='repeated_data_and_training.ini'
    config = ConfigParser()
    config.read(config_file)

    repeated_self_training(config_file, config.getint('SELF TRAINING', 'data_step'), config.getint('SELF TRAINING', 'runs'), config.getfloat('SELF TRAINING', 'win_rate'))

if __name__ == '__main__':
    _main()
