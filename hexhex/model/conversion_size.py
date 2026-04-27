import sys

import os

import torch

from hexhex.utils.paths import REFERENCE_MODELS_DIR, reference_model_path
from hexhex.utils.utils import device


def convert_boardsize_of_model(model_name, new_bs):
    new_bs = int(new_bs)
    checkpoint = torch.load(reference_model_path(model_name), map_location=device)
    config = checkpoint['config']
    config['board_size'] = new_bs

    bias_key = 'bias'
    while True:
        if bias_key in checkpoint['model_state_dict']:
            checkpoint['model_state_dict'][bias_key] = torch.zeros(new_bs**2)
            break
        bias_key = 'internal_model.' + bias_key

    out_path = os.path.join(REFERENCE_MODELS_DIR, f'{new_bs}_{model_name}.pt')
    torch.save({
    'model_state_dict': checkpoint['model_state_dict'],
    'config': config,
    'optimizer': False
    }, out_path)
    print('=== converted model size ===')
    print(f'wrote {out_path}')

if __name__ == '__main__':
    old_model_name = sys.argv[1]
    board_size = sys.argv[2]
    convert_boardsize_of_model(old_model_name, board_size)
