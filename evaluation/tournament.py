#!/usr/bin/env python3
import itertools

import torch
import device
import evaluate_two_models
from evaluation import elo
from evaluation.match import MatchResults


def play_tournament(model_files):
    models = [torch.load(model_file, map_location=device.device) for model_file in model_files]

    all_results = []

    for first_idx, second_idx in itertools.combinations(range(len(model_files)), 2):
        results, signed_chi_squared = evaluate_two_models.play_games(
                models=(models[first_idx], models[second_idx]),
                device=device.device,
                openings=False,
                number_of_games=32,
                batch_size=32,
                board_size=5,
                temperature=0.1,
                temperature_decay=0,
                plot_board=False
        )
        all_results.append(MatchResults(model_files[first_idx], model_files[second_idx], results))
    return all_results

# def test():
#     model_files = [f'models/5_gen{i}.pt' for i in range(0, 140, 10)] + ['models/4M_random_0.pt', 'models/five_board_wd0.001.pt', ]
#     tournament = play_tournament(model_files)
#     elo.create_ratings(tournament)
#
# test()

