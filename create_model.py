import torch
from hexconvolution import NoMCTSModel

import argparse
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')
parser = argparse.ArgumentParser()

parser.add_argument('--board_size', type=int, default=config.get('CREATE MODEL', 'board_size'))
parser.add_argument('--layers', type=int, default=config.get('CREATE MODEL', 'layers'))
parser.add_argument('--intermediate_channels', type=int, default=config.get('CREATE MODEL', 'intermediate_channels'))
parser.add_argument('--model_name', type=str, default=config.get('CREATE MODEL', 'model_name'))

args = parser.parse_args()

model = NoMCTSModel(board_size=args.board_size, layers=args.layers, intermediate_channels=args.intermediate_channels)
torch.save(model, f'models/{args.model_name}.pt')
