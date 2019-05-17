#!/usr/bin/env python3
import torch

import sys
from configparser import ConfigParser

import hex.training.train as train
import hex.elo.elo as elo
from hex.creation import create_data, create_model
from hex.evaluation import evaluate_two_models
from  hex.utils.utils import load_model


def league(config_file, champions, runs, chi_squared_test_statistic):
    """
    runs a self training loop
    each iteration produces a new model which is then trained on self-play data
    data and model names are looped, so they take limited space and discard old data
    a model is preferred over the last if it performs significantly better according to a chi squared test
    after a round of champions is trained a league is played between all current champions and old league winners
    the new league winner becomes the initial champion for the next round
    """
    config = ConfigParser()
    config.read(config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    champion, model_args = load_model(f'models/{config.get("CREATE DATA", "model")}.pt')
    champion_filename = f'{config.get("SELF TRAINING", "champion_names")}0'
    torch.save({
        'model_state_dict': champion.state_dict(),
        'board_size': model_args.board_size,
        'model_type': model_args.model_type,
        'layers': model_args.layers,
        'layer_type': model_args.layer_type,
        'intermediate_channels': model_args.intermediate_channels,
        'optimizer': False
        }, f'models/{champion_filename}.pt')
    print(f'wrote models/{champion_filename}.pt')

    data_range_min = config.getint('CREATE DATA', 'data_range_min')
    data_range_max = config.getint('CREATE DATA', 'data_range_max')
    data_pos = data_range_min

    champion_iter = 1
    champion_unbeaten_run = 0
    league_winners = 0
    league_winner_elo = 0

    for run in range(runs):
        champion, _ = load_model(f'models/{champion_filename}.pt')

        create_data.generate_data_files(
                file_counter_start=data_pos,
                file_counter_end=data_pos+1,
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

        if data_pos+1 == data_range_max:
            data_pos = data_range_min
        else:
            data_pos += 1

        train_args = train.get_args(config_file)
        champion_names = config.get("SELF TRAINING", "champion_names")
        new_model_name = f'{champion_names}{champion_iter}'

        train_args.load_model = champion_filename
        train_args.save_model = new_model_name
        if league_winners > 0:
            train_args.optimizer_load = True
        train.train(train_args)
        eval_args = evaluate_two_models.get_args(config_file)

        signed_chi_squared = evaluate_two_models.play_games(
                models=(load_model(f'models/{new_model_name}.pt')[0], champion),
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

            if champion_iter+1 == champions:
                champions_list = [f'{champion_names}{idx}' for idx in range(champions+league_winners)]
                results = elo.play_tournament(champions_list, eval_args)
                ratings = elo.create_ratings(results)
                champions_with_ratings = list(zip(ratings, champions_list))
                if league_winners > 0:
                    league_winner_elo -= champions_with_ratings[-1][0]
                champions_with_ratings.sort(reverse=True)

                league_winner_tuple = champions_with_ratings[0]
                league_winner_elo += league_winner_tuple[0]
                print(f'{league_winner_tuple[1]} won the league! It has ELO {league_winner_elo}!')
                league_winner, model_args = load_model(f'models/{league_winner_tuple[1]}.pt')
                champion_filename = f'{champion_names}{champions+league_winners}'
                torch.save({
                    'model_state_dict': league_winner.state_dict(),
                    'board_size': model_args.board_size,
                    'model_type': model_args.model_type,
                    'layers': model_args.layers,
                    'layer_type': model_args.layer_type,
                    'intermediate_channels': model_args.intermediate_channels,
                    'optimizer': torch.load(f'models/{league_winner_tuple[1]}.pt', map_location=device)['optimizer']
                    }, f'models/{champion_filename}.pt')
                print(f'wrote models/{champion_filename}.pt')
                league_winners += 1
                champion_iter = 0

            else:
                champion_iter += 1
                champion_filename = new_model_name
                print(f'Accept {champion_filename} as new champion!')

        else:
            champion_unbeaten_run += 1
            print(f'The champion remains in place, unbeaten for {champion_unbeaten_run} iterations. Iteration: {run+1}')

def _main():
    config_file=sys.argv[1]
    config = ConfigParser()
    config.read(config_file)

    league(config_file,
        config.getint('SELF TRAINING', 'champions'),
        config.getint('SELF TRAINING', 'runs'),
        config.getfloat('SELF TRAINING', 'chi_squared_test_statistic')
        )

if __name__ == '__main__':
    _main()
