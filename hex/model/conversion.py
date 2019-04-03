import torch
import sys

from hex.creation.create_model import get_args, create_model
from hex.utils.utils import device

def convert_model(config, old_model_file):
    args = get_args(config)
    new_model = create_model(args)
    old_model = torch.load(old_model_file, map_location=device)

    new_model_file = f'models/{args.model_name}.pt'
    torch.save({
        'model_state_dict': old_model.state_dict(),
        'board_size': args.board_size,
        'model_type': args.model_type,
        'layers': args.layers,
        'layer_type': args.layer_type,
        'intermediate_channels': args.intermediate_channels,
        'optimizer': False
        }, new_model_file)
    print('=== converted model ===')
    print(f'wrote {new_model_file}\n')
    return

if __name__ == '__main__':
    old_model_file = sys.argv[1]
    convert_model('config.ini', old_model_file)
