import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data.dataset import TensorDataset, ConcatDataset

import os
import argparse
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')
parser = argparse.ArgumentParser()

parser.add_argument('--load_model', type=str, default=config.get('TRAIN', 'load_model'))
parser.add_argument('--save_model', type=str, default=config.get('TRAIN', 'save_model'))
parser.add_argument('--data', type=str, default=config.get('TRAIN', 'data'))
parser.add_argument('--data_range_min', type=int, default=config.get('TRAIN', 'data_range_min'))
parser.add_argument('--data_range_max', type=int, default=config.get('TRAIN', 'data_range_max'))
parser.add_argument('--weight_decay', type=float, default=config.get('TRAIN', 'weight_decay'))
parser.add_argument('--batch_size', type=int, default=config.get('TRAIN', 'batch_size'))
parser.add_argument('--epochs', type=int, default=config.get('TRAIN', 'epochs'))

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_list = []
for idx in range(args.data_range_min, args.data_range_max):
    board_states, moves, targets = torch.load('data/{}_{}.pt'.format(args.data, idx))
    dataset_list.append(TensorDataset(board_states, moves, targets))
concat_dataset = ConcatDataset(dataset_list)
positionloader = torch.utils.data.DataLoader(concat_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

model = torch.load('models/{}.pt'.format(args.load_model))
model.to(device)
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adadelta(model.parameters(), weight_decay=args.weight_decay)

for epoch in range(args.epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, (board_states, moves, labels) in enumerate(positionloader, 0):
        # get the inputs
        board_states, moves, labels = board_states.to(device), moves.to(device), labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(board_states)
        output_values = torch.gather(outputs, 1, moves)

        loss = criterion(output_values.view(-1), labels)
        
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    
    l2loss = sum(torch.pow(p, 2).sum() for p in model.parameters() if p.requires_grad)
    print('[%d] pred_loss: %.3f l2_param_loss: %.3f' %(epoch + 1, running_loss, l2loss))
    torch.save(model, 'models/{}_{}.pt'.format(args.save_model, epoch))

print('Finished Training')

torch.save(model, 'models/{}.pt'.format(args.save_model))