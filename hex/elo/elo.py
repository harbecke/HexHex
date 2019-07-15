#!/usr/bin/env python3
import copy
import itertools
import math
from collections import defaultdict

import numpy as np

from hex.evaluation import evaluate_two_models
from hex.utils.utils import device, load_model


def play_tournament(model_list, args):
    """
    Plays tournament of all vs all and returns results as 2D dictionary.

    :return: result[first][second] gives the number of games that first has won against second
    """
    num_models = len(model_list)
    models = [load_model(f'models/{model_file}.pt') for model_file in model_list]
    all_results = defaultdict(lambda: defaultdict(int))

    for first_idx, second_idx in itertools.combinations(range(num_models), 2):
        result, signed_chi_squared = evaluate_two_models.play_games(
                models=(models[first_idx], models[second_idx]),
                device=device,
                openings=args.getboolean('openings'),
                number_of_games=args.getint('number_of_games'),
                batch_size=args.getint('batch_size'),
                temperature=args.getfloat('temperature'),
                temperature_decay=args.getfloat('temperature_decay'),
                plot_board=args.getboolean('plot_board')
        )
        all_results[first_idx][second_idx] = result[0][0] + result[1][0]
        all_results[second_idx][first_idx] = result[0][1] + result[1][1]

    return all_results


def add_to_tournament(model_list, new_model_name, args, old_results):
    """
    Adds new_model to existing tournament by playing against all other teams.
    """

    if old_results is None:
        new_results = defaultdict(lambda: defaultdict(int))
    else:
        new_results = copy.deepcopy(old_results)

    new_index = len(model_list)

    sub_model_ids = list(np.random.choice(range(len(model_list)),
                                          size=min(len(model_list), args.getint('max_num_opponents', fallback=10)),
                                          replace=False))
    models = [load_model(f'models/{model_file}.pt') for model_file in model_list]
    new_model = load_model(f'models/{new_model_name}.pt')

    for old_index in sub_model_ids:
        old_model = models[old_index]
        result, signed_chi_squared = evaluate_two_models.play_games(
                models=(old_model, new_model),
                device=device,
                openings=args.getboolean('openings'),
                number_of_games=args.getint('number_of_games'),
                batch_size=args.getint('batch_size'),
                temperature=args.getfloat('temperature'),
                temperature_decay=args.getfloat('temperature_decay'),
                plot_board=args.getboolean('plot_board')
        )

        new_results[old_index][new_index] = result[0][0] + result[1][0]
        new_results[new_index][old_index] = result[0][1] + result[1][1]

    return new_results


def create_ratings(results, runs=100):
    # from https://en.wikipedia.org/wiki/Bradley-Terry_model

    num_models = len(results)
    # + 0.001 for numerical reasons
    results_sum = [sum(results[x][y] + 0.01 for y in range(num_models)) for x in range(num_models)]
    games = 2*sum(results_sum)/num_models
    p_list = results_sum[:]

    for _ in range(runs):
        inverse_p_list = [[1/(p_list[idx1]+p_list[idx2])
                           for idx2 in range(num_models) if idx1 != idx2] for idx1 in range(num_models)]
        sum_inverse_p_list = [1/sum(i) for i in inverse_p_list]
        new_p_list = [results_sum[idx]/games*sum_inverse_p_list[idx] for idx in range(num_models)]
        sum_p_list = sum(new_p_list)
        p_list = [p/sum_p_list for p in new_p_list]

    min_value = min(p_list)
    elo_ratings = [math.log10(p/min_value)*400 for p in p_list]

    return elo_ratings

