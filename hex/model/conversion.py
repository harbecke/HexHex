import sys
from configparser import ConfigParser

import torch

from hex.creation.create_model import create_and_store_model
from hex.utils.utils import device


def convert_model(model_name):
    old_model = torch.load(f'models/{model_name}.pt', map_location=device)
    config = ConfigParser()
    config = config['DEFAULT']
    for key, val in old_model.items():
        if key != 'model_state_dict':
            config[key] = str(val)
    config['model_name'] = model_name
    create_and_store_model(config)
    print('=== converted model ===')
    return


def _main():
    old_model_name = sys.argv[1]
    convert_model(old_model_name)


if __name__ == '__main__':
    _main()