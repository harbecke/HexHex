#!/usr/bin/env python3

import torch
from hex.model.hexconvolution import NoMCTSModel, RandomModel, MCTSModel, InceptionModel

import argparse
from configparser import ConfigParser

def get_args(config_file):
    config = ConfigParser()
    config.read(config_file)
    parser = argparse.ArgumentParser()

    parser.add_argument('--board_size', type=int, default=config.getint('CREATE MODEL', 'board_size'))
    parser.add_argument('--model_type', type=str, default=config.get('CREATE MODEL', 'model_type'))
    parser.add_argument('--layer_type', type=str, default=config.get('CREATE MODEL', 'layer_type'))
    parser.add_argument('--layers', type=int, default=config.getint('CREATE MODEL', 'layers'))
    parser.add_argument('--intermediate_channels', type=int, default=config.getint('CREATE MODEL', 'intermediate_channels'))
    parser.add_argument('--model_name', type=str, default=config.get('CREATE MODEL', 'model_name'))

    return parser.parse_args(args=[])

def create_model(args):
    print("=== creating model ===")
    if args.model_type == 'random':
        model = RandomModel(board_size=args.board_size)
    elif args.model_type == 'mcts':
        model = MCTSModel(board_size=args.board_size, layers=args.layers,
                          intermediate_channels=args.intermediate_channels, skip_layer=args.layer_type)
    elif args.model_type == 'inception':
        model = InceptionModel(board_size=args.board_size, layers=args.layers, 
            intermediate_channels=args.intermediate_channels)
    else:
        model = NoMCTSModel(board_size=args.board_size, layers=args.layers,
                        intermediate_channels=args.intermediate_channels, skip_layer=args.layer_type)
    return model

def create_model_from_args(args):
    model = create_model(args)
    model_file = f'models/{args.model_name}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'board_size': args.board_size,
        'model_type': args.model_type,
        'layers': args.layers,
        'layer_type': args.layer_type,
        'intermediate_channels': args.intermediate_channels,
        'optimizer': False
        }, model_file)
    print(f'wrote {model_file}\n')

if __name__ == '__main__':
    args = get_args('config.ini')
    create_model_from_config_file(args)
