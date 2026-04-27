#!/usr/bin/env python3
import copy
import math
from collections import defaultdict

from hexhex.evaluation import evaluate_two_models
from hexhex.logic import temperature
from hexhex.utils.paths import run_model_path
from hexhex.utils.utils import load_model


def add_to_tournament(model_list, new_model_name, cfg, old_results):
    """
    Adds new_model to existing tournament by playing against all other teams.
    """

    if old_results is None:
        new_results = defaultdict(lambda: defaultdict(int))
    else:
        new_results = copy.deepcopy(old_results)

    sub_model_names = model_list[:cfg.max_num_opponents]
    new_model = load_model(run_model_path(new_model_name))

    for old_model_file in sub_model_names:
        old_model = load_model(run_model_path(old_model_file))
        result, signed_chi_squared = evaluate_two_models.play_games(
                models=(old_model, new_model),
                num_opened_moves=cfg.num_opened_moves,
                number_of_games=cfg.number_of_games,
                batch_size=cfg.batch_size,
                temperature_schedule=temperature.from_config(cfg.temperature),
                plot_board=cfg.plot_board
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
