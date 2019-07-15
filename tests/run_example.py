#!/usr/bin/env python3

from hex.creation import create_data, create_model
from hex.evaluation import evaluate_two_models
from hex.training import train


def _main(config_file = 'config.ini'):
    '''
    executes all basic scripts with values from sample config
    should be run as test before every commit
    '''
    args = create_model.get_args(config_file)
    create_model.create_and_store_model(args)
    create_data.main(config_file)
    train.train_by_config_file(config_file)
    evaluate_two_models.evaluate(config_file)


if __name__ == "__main__":
    _main()
