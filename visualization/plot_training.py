#!/usr/bin/env python3
from collections import defaultdict

import csv
import matplotlib.pyplot as plt

def read_data(log_file: str):
    with open(log_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')

        idx_to_name = [
            'batch',
            'val_loss',
            'pred_loss',
            'weighted_param_loss'
        ]

        columns = defaultdict(lambda : [])

        for row in reader:
            if '#' in row[0]:
                continue
            for idx, item in enumerate(row):
                columns[idx_to_name[idx]].append(float(item))

        return columns


def plot_training(data):
    fig, (ax) = plt.subplots(1, 1, figsize=(5, 5))
    print(data['val_loss'])
    ax.scatter(data['batch'], data['val_loss'])
    ax.set_xlabel('# batch')
    ax.set_ylabel('val_loss')
    plt.show()

def main():
    data = read_data('../logs/train.csv')
    plot_training(data)


if __name__ == '__main__':
    main()
