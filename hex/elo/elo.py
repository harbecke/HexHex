#!/usr/bin/env python3
import itertools
import torch
import math

from hex.utils.utils import device, load_model
from hex.evaluation import evaluate_two_models


def play_tournament(model_list):

    num_models = len(model_list)
    models = [load_model(f'models/{model_file}.pt') for model_file in model_list]
    all_results = [[0 for _ in range(num_models)] for _ in range(num_models)]

    for first_idx, second_idx in itertools.combinations(range(num_models), 2):
        result, signed_chi_squared = evaluate_two_models.play_games(
                models=(models[first_idx], models[second_idx]),
                device=device,
                openings=False,
                number_of_games=32,
                batch_size=32,
                board_size=5,
                temperature=0.1,
                temperature_decay=0,
                plot_board=False
        )
        all_results[first_idx][second_idx] = result[0][0] + result[1][0]
        all_results[second_idx][first_idx] = result[0][1] + result[1][1]

    return all_results


def create_ratings(results, runs=100):
    #from https://en.wikipedia.org/wiki/Bradley-Terry_model

    num_models = len(results)
    results_sum = [sum(result)+1 for result in results]
    games = 2*sum(results_sum)/num_models
    p_list = results_sum[:]

    for _ in range(runs):
        inverse_p_list = [[1/(p_list[idx1]+p_list[idx2]) for idx2 in range(num_models) if idx1!=idx2] for idx1 in range(num_models)]
        sum_inverse_p_list = [1/sum(i) for i in inverse_p_list]
        new_p_list = [results_sum[idx]/games*sum_inverse_p_list[idx] for idx in range(num_models)]
        sum_p_list = sum(new_p_list)
        p_list = [p/sum_p_list for p in new_p_list]

    min_value = min(p_list)
    elo_ratings = [math.log10(p/min_value)*400 for p in p_list]

    return elo_ratings


def output_ratings(model_list, output_file='ratings.txt'):
    
    tournament = play_tournament(model_list)
    ratings = create_ratings(tournament)
    model_with_ratings = list(zip(ratings, model_list))
    model_with_ratings.sort(reverse=True)

    with open(output_file, 'w') as file:
        file.write('ELO\t\tModel\n')
        for rating, model in model_with_ratings:
            file.write('{}\t\t{}\n'.format(int(rating), model))
