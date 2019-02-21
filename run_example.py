#!/usr/bin/env python3

import create_data
import create_model
import evaluate_two_models
import train


def _main(config_file = 'sample_config.ini'):
    '''
    executes all basic scripts with values from sample config
    should be run as test before every commit
    '''
    create_model.create_model_from_config_file(config_file)
    create_data.main(config_file)
    train.train_by_config_file(config_file)
    evaluate_two_models.evaluate(config_file)


if __name__ == "__main__":
    _main()
