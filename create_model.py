import torch
from hexconvolution import NoMCTSModel

import argparse
from configparser import ConfigParser

def get_args(config_file):
    config = ConfigParser()
    config.read(config_file)
    parser = argparse.ArgumentParser()

    parser.add_argument('--board_size', type=int, default=config.get('CREATE MODEL', 'board_size'))
    parser.add_argument('--layers', type=int, default=config.get('CREATE MODEL', 'layers'))
    parser.add_argument('--intermediate_channels', type=int, default=config.get('CREATE MODEL', 'intermediate_channels'))
    parser.add_argument('--model_name', type=str, default=config.get('CREATE MODEL', 'model_name'))

    return parser.parse_args()

def create_model(args):
    print("=== creating model ===")
    model = NoMCTSModel(board_size=args.board_size, layers=args.layers,
                        intermediate_channels=args.intermediate_channels)
    model_file = f'models/{args.model_name}.pt'
    torch.save(model, model_file)
    print(f'wrote {model_file}\n')

def create_model_from_config_file(config_file):
    args = get_args(config_file)
    create_model(args)

if __name__ == '__main__':
    create_model_from_config_file('config.ini')