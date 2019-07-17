#!/usr/bin/env python3

import torch

from hex.model import hexconvolution
from hex.utils.logger import logger


def create_model(config):
    board_size = config.getint('board_size')
    model_type = config['model_type']
    rotation_model = config.getboolean('rotation_model')
    vertical_model = config.getboolean('vertical_model')

    if model_type == 'random':
        model = hexconvolution.RandomModel(board_size=board_size)
    elif model_type == 'noswitch':
        model = hexconvolution.NoSwitchModel(
            board_size=board_size,
            layers=config.getint('layers'),
            intermediate_channels=config.getint('intermediate_channels')
        )
    elif model_type == 'inception':
        model = hexconvolution.InceptionModel(
            board_size=board_size,
            layers=config.getint('layers'),
            intermediate_channels=config.getint('intermediate_channels')
        )
    elif model_type == 'nomcts':
        model = hexconvolution.NoMCTSModel(
            board_size=board_size,
            layers=config.getint('layers'),
            intermediate_channels=config.getint('intermediate_channels'),
            skip_layer=config.get('layer_type')
        )
    else:
        logger.error(f"Unknown model_type: {model_type}")
        exit(1)

    if rotation_model == True:
        model = hexconvolution.RotationWrapperModel(model)

    if vertical_model == True:
        model = hexconvolution.VerticalWrapperModel(model)

    return model


def create_and_store_model(config, name):
    model = create_model(config)
    model_file = f'models/{name}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'optimizer': False
        }, model_file)
    logger.info(f'wrote {model_file}\n')
