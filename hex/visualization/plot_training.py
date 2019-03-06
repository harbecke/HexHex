#!/usr/bin/env python3

import argparse
import csv
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_files', metavar='N', type=str, nargs='+',
                        help='log files from which training records are gathered')
    return parser.parse_args()


def read_data_single_file(log_file):
    with open(log_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')

        idx_to_name = [
            'batch',
            'val_loss',
            'pred_loss',
            'weighted_param_loss',
            'running_time',
        ]

        columns = {'name': log_file}
        for name in idx_to_name:
            columns[name] = []

        for row in reader:
            if '#' in row[0]:
                continue
            for idx, item in enumerate(row):
                columns[idx_to_name[idx]].append(float(item))

        return columns


def read_data(log_files):
    return [read_data_single_file(log_file) for log_file in log_files]


def plot_training(data):
    fig, (ax) = plt.subplots(1, 1, figsize=(5, 5))
    x = 'batch'
    y = 'val_loss'
    for run in data:
        ax.plot(run[x], run[y], label=run['name'])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    plt.show()


def main():
    args = get_args()
    data = read_data(args.log_files)
    plot_training(data)


if __name__ == '__main__':
    main()
