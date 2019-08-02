#!/usr/bin/env python3
import copy
import math
from collections import defaultdict

import numpy as np

from hex.evaluation import evaluate_two_models
from hex.utils.utils import load_model


def add_to_tournament(model_list, new_model_name, args, old_results):
    """
    Adds new_model to existing tournament by playing against all other teams.
    """

    if old_results is None:
        new_results = defaultdict(lambda: defaultdict(int))
    else:
        new_results = copy.deepcopy(old_results)

    sub_model_names = list(np.random.choice(model_list,
                                            size=min(len(model_list), args.getint('max_num_opponents', fallback=10)),
                                            replace=False))
    new_model = load_model(f'models/{new_model_name}.pt')

    for old_model_file in sub_model_names:
        old_model = load_model(f'models/{old_model_file}.pt')
        result, signed_chi_squared = evaluate_two_models.play_games(
                models=(old_model, new_model),
                openings=args.getboolean('openings'),
                number_of_games=args.getint('number_of_games'),
                batch_size=args.getint('batch_size'),
                temperature=args.getfloat('temperature'),
                temperature_decay=args.getfloat('temperature_decay'),
                plot_board=args.getboolean('plot_board')
        )

        new_results[old_model_file][new_model_name] = result[0][0] + result[1][0]
        new_results[new_model_name][old_model_file] = result[0][1] + result[1][1]

    return new_results


def create_ratings(results, runs=100):
    # from https://en.wikipedia.org/wiki/Bradley-Terry_model

    all_models = set(results.keys())
    for _, value in results.items():
        for v in value.keys():
            all_models.add(v)

    # + 0.01 for numerical reasons
    results_sum = {x: sum(results[x][y] + 0.01 for y in all_models) for x in all_models}
    p_list = results_sum.copy()

    for _ in range(runs):
        inverse_p_list = {idx1: {idx2: (results[idx1][idx2]+results[idx2][idx1] + 0.01)/(p_list[idx1]+p_list[idx2])
                           for idx2 in all_models if idx1 != idx2} for idx1 in all_models}
        new_p_list = {idx: results_sum[idx]/sum(inverse_p_list[idx].values()) for idx in all_models}
        sum_p_list = sum(new_p_list.values())
        p_list = {p: new_p_list[p]/sum_p_list for p in new_p_list}

    min_value = p_list[list(results.keys())[0]]
    elo_ratings = {p: math.log10(p_list[p]/min_value)*400 for p in p_list}

    return elo_ratings
