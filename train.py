import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data.dataset import TensorDataset, ConcatDataset

import os
import argparse
from configparser import ConfigParser


def validation(validation_data, model, criterion, device):
    board_tensor, moves_tensor, target_tensor = torch.load(f'data/{validation_data}.pt')
    board_tensor, moves_tensor, target_tensor = board_tensor.to(device), moves_tensor.to(device), target_tensor.to(device)

    
    return loss


def train_model(model, save_model_path, dataloader, criterion, optimizer, epochs, device, save_frequency='epoch', print_loss_frequency=100, validation_triple=False):

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, (board_states, moves, labels) in enumerate(dataloader, 0):
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
            running_loss += loss.item()
            
            if i % print_loss_frequency == 0:
                l2loss = sum(torch.pow(p, 2).sum() for p in model.parameters() if p.requires_grad)
                if validation_triple:
                    with torch.no_grad():
                        val_pred_tensor = model(validation_triple[0])
                        val_values = torch.gather(val_pred_tensor, 1, validation_triple[1])
                        val_loss = criterion(val_values.view(-1), validation_triple[2])
                    print('val_loss: %.3f  pred_loss: %.3f  l2_param_loss: %.3f'%(val_loss, loss.item(), l2loss))
                else:
                    print('pred_loss: %.3f  l2_param_loss: %.3f'%(loss.item(), l2loss))


        l2loss = sum(torch.pow(p, 2).sum() for p in model.parameters() if p.requires_grad)
        print('Epoch [%d] pred_loss: %.3f l2_param_loss: %.3f' %(epoch + 1, running_loss, l2loss))
        if save_frequency == 'epoch':
            torch.save(model, 'models/{}_{}.pt'.format(save_model_path, epoch))

    print('Finished Training')
    if save_frequency == 'once':
        torch.save(model, 'models/{}.pt'.format(save_model_path))

def main():
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
    parser.add_argument('--validation_data', type=str, default=config.get('TRAIN', 'validation_data'))
    parser.add_argument('--validation_bool', type=bool, default=config.get('TRAIN', 'validation_bool'))
    parser.add_argument('--save_frequency', type=str, default=config.get('TRAIN', 'save_frequency'))
    parser.add_argument('--print_loss_frequency', type=int, default=config.get('TRAIN', 'print_loss_frequency'))

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

    if args.validation_bool:
        val_board_tensor, val_moves_tensor, val_target_tensor = torch.load(f'data/{args.validation_data}.pt')
        val_triple = (val_board_tensor.to(device), val_moves_tensor.to(device), val_target_tensor.to(device))
        train_model(model, args.save_model, positionloader, criterion, optimizer, args.epochs, device, 'epoch', args.print_loss_frequency, val_triple)
    else:
        train_model(model, args.save_model, positionloader, criterion, optimizer, args.epochs, device, 'epoch', args.print_loss_frequency)

if __name__ == "__main__":
    main()
