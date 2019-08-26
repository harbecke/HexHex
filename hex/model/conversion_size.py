import sys

import torch
from hex.utils.utils import device

def convert_boardsize_of_model(model_name, new_bs):
    checkpoint = torch.load(f'models/{model_name}.pt', map_location=device)
    config = checkpoint['config']
    config['board_size'] = new_bs

    bias_key = 'bias'
    while True:
        if bias_key in checkpoint['model_state_dict']:
            checkpoint['model_state_dict'][bias_key] = torch.zeros(int(new_bs)**2)
            break
        bias_key = 'internal_model.' + bias_key

    torch.save({
    'model_state_dict': checkpoint['model_state_dict'],
    'config': config,
    'optimizer': False
    }, f'models/{new_bs}_{model_name}.pt')
    print('=== converted model size ===')
    print(f'wrote models/{new_bs}_{model_name}.pt')

if __name__ == '__main__':
    old_model_name = sys.argv[1]
    board_size = sys.argv[2]
    convert_boardsize_of_model(old_model_name, board_size)
