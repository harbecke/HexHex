#!/usr/bin/env python3

import torch

from hex.model import hexconvolution
from hex.utils.logger import logger


def create_model(config):
    board_size = config.getint('board_size')
    model_type = config['model_type']
    switch_model = config.getboolean('switch_model')
    rotation_model = config.getboolean('rotation_model')

    if model_type == 'random':
        model = hexconvolution.RandomModel(board_size=board_size)
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
    elif model_type == 'conv':
        model = hexconvolution.Conv(
            board_size=board_size,
            layers=config.getint('layers'),
            intermediate_channels=config.getint('intermediate_channels'),
            reach=config.getint('reach')
        )
    else:
        logger.error(f"Unknown model_type: {model_type}")
        exit(1)

    if switch_model == False:
        model = hexconvolution.NoSwitchWrapperModel(model)

    if rotation_model == True:
        model = hexconvolution.RotationWrapperModel(model)

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
