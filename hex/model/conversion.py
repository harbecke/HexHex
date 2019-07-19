import sys
from configparser import ConfigParser

import torch
import copy
import torch.nn as nn

from hex.creation.create_model import create_and_store_model, create_model
from hex.utils.utils import device


def convert_model(model_name):
    old_model = torch.load(f'models/{model_name}.pt', map_location=device)
    config = ConfigParser()
    config = config['DEFAULT']
    for key, val in old_model.items():
        if key != 'model_state_dict':
            config[key] = str(val)
    create_and_store_model(config, model_name)
    print('=== converted model ===')
    return


def convert_boardsize_of_model(model_name, new_bs):
    checkpoint = torch.load(f'models/{model_name}.pt', map_location=device)
    model = create_model(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    init_model = model
    config = checkpoint['config']
    bs = config.getint('board_size')
    assert config['model_type'] == 'nomcts'
    assert config['layer_type'] != 'star'
    assert new_bs >= bs
    config['board_size'] = str(new_bs)
    new_model = create_model(config)
    init_new_model = new_model
    while hasattr(model, 'internal_model'):
        model = model.internal_model
    while hasattr(new_model, 'internal_model'):
        new_model = new_model.internal_model
    new_model.conv = copy.deepcopy(model.conv)
    new_model.skiplayers = nn.ModuleList([copy.deepcopy(layer) for layer in model.skiplayers])
    new_model.policyconv = copy.deepcopy(model.policyconv)
    new_model.policybn = copy.deepcopy(model.policybn)
    for index in range(init_new_model.board_size ** 2):
        new_model.policylin.bias[index] = model.policylin.bias[getindex_small_board(index, bs, new_bs)]
    print(new_model.policylin.bias)
    for i1 in range(init_new_model.board_size ** 2):
        for i2 in range(init_new_model.board_size ** 2):
            i1_shifted = getindex_small_board(i1, bs, new_bs)
            i2_shifted = getindex_small_board_shifted(i1, i2, bs, new_bs)
            for n in range(new_model.policy_channels):
                if i2_shifted != -1:
                    new_model.policylin.weight[i1, i2 + n * new_bs ** 2] = model.policylin.weight[i1_shifted][
                        i2_shifted + n * bs ** 2]
                else:
                    new_model.policylin.weight[i1, i2 + n * new_bs ** 2] = 0.
    checkpoint['config'] = config
    checkpoint['model_state_dict'] = init_new_model.state_dict()
    checkpoint['optimizer'] = None
    file_name = f'models/{model_name}_rs{new_bs}.pt'
    torch.save(checkpoint, file_name)
    print('=== converted model ===')
    return


def getindex_small_board(index, bs1, bs2):
    x, y = index%bs2, index//bs2
    r = (bs1+1)//2
    if bs2 - x < r:
        x = bs1 - bs2 + x
    elif x > r:
        x = r
    if bs2 - y < r:
        y = bs1 - bs2 + y
    elif y > r:
        y = r
    return bs1 * y + x

def getindex_small_board_shifted(index1, index2, bs1, bs2):
    dx, dy = index2%bs2 - index1%bs2, index2//bs2-index1//bs2
    x_new, y_new = getindex_small_board(index1, bs1, bs2)%bs1, getindex_small_board(index1, bs1, bs2)//bs1
    if bs1 > x_new + dx >= 0 and bs1 > y_new + dy >= 0:
        return (y_new+dy)*bs1 + x_new+dx
    else:
        return -1


def _main():
    old_model_name = sys.argv[1]
    convert_boardsize_of_model(old_model_name, 9)


if __name__ == '__main__':
    _main()