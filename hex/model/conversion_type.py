import sys

import torch
from hex.utils.utils import device


def convert_model(model_name):
    checkpoint = torch.load(f'models/{model_name}.pt', map_location=device)
    config = checkpoint['config']
    if config['model_type'] in ['inception', 'conv']:
        config['model_type'] = 'conv'

        weight_key = 'conv.weight'
        while True:
            if weight_key in checkpoint['model_state_dict']:
                config['reach'] = str(checkpoint['model_state_dict'][weight_key].shape[2] // 2)
                break
            weight_key = 'internal_model.' + weight_key

        torch.save({
        'model_state_dict': checkpoint['model_state_dict'],
        'config': config,
        'optimizer': False
        }, f'models/{model_name}.pt')
        print('=== converted model type ===')
    else:
        print('=== model type is not "inception" or "conv" ===')
    return

if __name__ == '__main__':
    old_model_name = sys.argv[1]
    convert_model(old_model_name)
