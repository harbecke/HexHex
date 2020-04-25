#!/usr/bin/env python3

import torch

from hexhex.model import hexconvolution
from hexhex.utils.logger import logger


def create_model(config, export_mode=False):
    board_size = config.getint('board_size')
    switch_model = config.getboolean('switch_model')
    rotation_model = config.getboolean('rotation_model')

    model = hexconvolution.Conv(
        board_size=board_size,
        layers=config.getint('layers'),
        intermediate_channels=config.getint('intermediate_channels'),
        reach=config.getint('reach'),
        export_mode=export_mode
        )

    if not switch_model:
        model = hexconvolution.NoSwitchWrapperModel(model)

    if rotation_model:
        model = hexconvolution.RotationWrapperModel(model, export_mode)

    return model


def create_and_store_model(config, name):
    model = create_model(config)
    model_file = f'models/{name}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
        }, model_file)
    logger.info(f'wrote {model_file}\n')
