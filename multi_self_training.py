#!/usr/bin/env python3
import torch

from torch.distributions.log_normal import LogNormal
from torch.distributions.binomial import Binomial

import train

from argparse import ArgumentParser
from configparser import ConfigParser

from random import getrandbits
from time import time, gmtime, strftime

def get_model_parameters(weight_decay_mean, weight_decay_std, batch_size_np, batch_size_p, learning_rate_mean, learning_rate_std):

    weight_decay = float(LogNormal(-9, 2).sample())
    batch_size = 2**int(Binomial(10, 0.5).sample())
    learning_rate = float(LogNormal(-4.5, 1).sample())
    if not getrandbits(1):
        optimizer = 'adadelta'
    else:
        optimizer = 'adam'

    return weight_decay, batch_size, learning_rate, optimizer


def multi_self_training(starting_models, number_of_runs, model_csv, weight_decay_mean, weight_decay_std, batch_size_np,
        batch_size_p, learning_rate_mean, learning_rate_std, config_file):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    doc_time = strftime("%Y%m%d_%H%M%S", gmtime())

    for idx in range(number_of_runs):
        for model_name in starting_models:
            time_str = strftime("%Y%m%d_%H%M%S", gmtime())

            train_args = train.get_args(config_file)
            weight_decay, batch_size, learning_rate, optimizer = get_model_parameters(weight_decay_mean, weight_decay_std,
                batch_size_np, batch_size_p, learning_rate_mean, learning_rate_std)

            train_args.load_model = model_name
            train_args.save_model = model_name + time_str
            train_args.weight_decay = weight_decay
            train_args.batch_size = batch_size
            train_args.optimizer = optimizer
            train_args.learning_rate = learning_rate

            start = time()
            train.train(train_args)
            duration = str(int(time() - start))
            with open(model_csv+doc_time+'.csv', 'a') as file:
                file.write(model_name+time_str+','+duration+','+str(weight_decay)+','+str(batch_size)+','+str(optimizer)+','+str(learning_rate)+'\n')


def main(config_file = 'config.ini'):

    config = ConfigParser()
    config.read(config_file)
    
    starting_models = [model for model in config.get('MULTI SELF TRAINING', 'starting_models').split(",")]
    number_of_runs = config.getint('MULTI SELF TRAINING', 'number_of_runs')
    model_csv = config.get('MULTI SELF TRAINING', 'model_csv')

    weight_decay_mean = config.getfloat('MULTI SELF TRAINING', 'weight_decay_mean')
    weight_decay_std = config.getfloat('MULTI SELF TRAINING', 'weight_decay_std')
    batch_size_np = config.getint('MULTI SELF TRAINING', 'batch_size_np')
    batch_size_p = config.getfloat('MULTI SELF TRAINING', 'batch_size_p')
    learning_rate_mean = config.getfloat('MULTI SELF TRAINING', 'learning_rate_mean')
    learning_rate_std = config.getfloat('MULTI SELF TRAINING', 'learning_rate_std')
 
    multi_self_training(starting_models, number_of_runs, model_csv, weight_decay_mean, weight_decay_std, batch_size_np,
        batch_size_p, learning_rate_mean, learning_rate_std, config_file)

if __name__ == '__main__':
    main()
