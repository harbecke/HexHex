import csv
from configparser import ConfigParser

import torch

from hex.evaluation.evaluate_two_models import get_args, play_games
from hex.utils.utils import load_model


def multi_evaluate(config_file = 'config.ini'):
    print("=== multi evaluate models ===")

    config = ConfigParser()
    config.read(config_file)
    model_csv = config.get('MULTI EVALUATE MODEL', 'model_csv')
    output_csv = config.get('MULTI EVALUATE MODEL', 'output_csv')

    args = get_args(config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    results = []

    with open(model_csv+'.csv') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            models.append(row[0])

    for idx1 in range(len(models)):
        results.append([])
        for idx2 in range(len(models)):

            if idx1 < idx2:
                model1 = load_model(f'models/{args.model1}.pt')
                model2 = load_model(f'models/{args.model2}.pt')
                
                results[idx1].append(play_games(
                        models=(model1, model2),
                        openings=args.openings,
                        number_of_games=args.number_of_games,
                        batch_size=args.batch_size,
                        temperature=args.temperature,
                        temperature_decay=args.temperature_decay,
                        plot_board=args.plot_board)[1])

                with open(output_csv+'.csv', 'a') as csvfile:
                    csvfile.write(str(results[idx1][idx2-idx1-1])+',')

            elif idx1 == idx2:
                with open(output_csv+'.csv', 'a') as csvfile:
                    csvfile.write('0,')

            else:
                with open(output_csv+'.csv', 'a') as csvfile:
                    csvfile.write(str(-results[idx2][idx1-idx2-1])+',')

        with open(output_csv+'.csv', 'a') as csvfile:
            csvfile.write('\n')

    return results


if __name__ == '__main__':
    multi_evaluate()