#!/usr/bin/env python3
import torch

import sys
from configparser import ConfigParser

import hex.training.train as train
from hex.creation import create_data
from hex.evaluation import evaluate_two_models


def repeated_self_training(config_file, champions, runs, chi_squared_test_statistic):
    """
    runs a self training loop
    each iteration produces a new model which is then trained on self-play data
    data and model names are loop, so they take limited space and discard old data
    a model is preferred over the last if it performs significantly better according to a chi squared test
    """
    config = ConfigParser()
    config.read(config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    champion_iter = 0
    champion_unbeaten_run = 0
    champion_filename = config.get('CREATE DATA', 'model')
    data_range_min=config.getint('CREATE DATA', 'data_range_min')
    data_range_max=config.getint('CREATE DATA', 'data_range_max')
    data_range=[data_range_min, data_range_min+1]
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
                noise = config.get('CREATE DATA', 'noise'),
                noise_parameters = [float(parameter) for parameter in config.get('CREATE DATA', 'noise_parameters').split(",")],
                temperature=config.getfloat('CREATE DATA', 'temperature'),
                temperature_decay=config.getfloat('CREATE DATA', 'temperature_decay'),
                board_size=config.getint('CREATE DATA', 'board_size')
        )
        new_data_range_max = data_range[1]+1
        if data_range[1]+1 > data_range_max:
            data_range=[data_range_min, data_range_min+1]
        else:
            data_range=[data_range[0]+1, data_range[1]+1]

        train_args = train.get_args(config_file)
        new_model_name = f'{config.get("SELF TRAINING", "champion_names")}{champion_iter}'

        train_args.load_model = champion_filename
        train_args.save_model = new_model_name
        train.train(train_args)
        eval_args = evaluate_two_models.get_args(config_file)
        signed_chi_squared = evaluate_two_models.play_games(
                models=(torch.load('models/'+new_model_name+'.pt'), champion),
                openings=eval_args.openings,
                number_of_games=eval_args.number_of_games,
                device=device,
                batch_size=eval_args.batch_size,
                board_size=eval_args.board_size,
                temperature=eval_args.temperature,
                temperature_decay=eval_args.temperature_decay,
                plot_board=eval_args.plot_board)[1]

        if signed_chi_squared > chi_squared_test_statistic:
            champion_unbeaten_run = 1
            champion_filename = new_model_name
            champion_iter = (champion_iter+1)%champions
            print(f'Accept {champion_filename} as new champion!')
        else:
            if champion_unbeaten_run >= data_range_max-data_range_min:
                champion_unbeaten_run = 1
            else:
                champion_unbeaten_run += 1
            print(f'The champion remains in place, unbeaten for {champion_unbeaten_run} iterations. Iteration: {run+1}')

def _main():
    config_file=sys.argv[1]
    config = ConfigParser()
    config.read(config_file)

    repeated_self_training(config_file, config.getint('SELF TRAINING', 'champions'), config.getint('SELF TRAINING', 'runs'), config.getfloat('SELF TRAINING', 'chi_squared_test_statistic'))

if __name__ == '__main__':
    _main()
