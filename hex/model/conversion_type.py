import sys

import torch
from hex.utils.utils import device


def convert_model(model_name):
    checkpoint = torch.load(f'models/{model_name}.pt', map_location=device)
    config = checkpoint['config']
    if config['model_type'] == 'inception':
        config['model_type'] = 'conv'
        config['reach'] = config['board_size']
        torch.save({
        'model_state_dict': checkpoint['model_state_dict'],
        'config': config,
        'optimizer': False
        }, f'models/{model_name}.pt')
        print('=== converted model type ===')
    return

if __name__ == '__main__':
    old_model_name = sys.argv[1]
    convert_model(old_model_name)
