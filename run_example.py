#!/usr/bin/env python3

import create_data
import create_model
import evaluate_two_models
import train


def _main(config_file = 'sample_config.ini'):
    create_model.create_model(config_file)
    create_data.main(config_file)
    train.main(config_file)
    evaluate_two_models.evaluate(config_file)


if __name__ == "__main__":
    _main()
