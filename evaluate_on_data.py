import torch
import torch.nn as nn

import argparse
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default=config.get('EVALUATE DATA', 'model'))
parser.add_argument('--data', type=str, default=config.get('EVALUATE DATA', 'data'))
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss(reduction='sum')

model = torch.load('models/{}.pt'.format(args.model), map_location=device)
model = nn.DataParallel(model).to(device)

board_tensor, moves_tensor, target_tensor = torch.load('data/{}.pt'.format(data))
board_tensor, moves_tensor, target_tensor = board_tensor.to(device), moves_tensor.to(device), target_tensor.to(device)

with torch.no_grad():
    full_pred_tensor = model(board_tensor)
    local_predictions = torch.gather(full_pred_tensor, 1, moves_tensor)
    loss = criterion(local_predictions.view(-1), target_tensor)
    print(loss)
