import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical
from noise import singh_maddala_onto_output
from utils import zip_list_of_lists_first_dim_reversed


def tempered_moves_selection(output_tensor, temperature):
    if temperature == 0:
        return output_tensor.argmax(1)
    else:
        normalized_output_tensor = output_tensor/output_tensor.max(1)[0].unsqueeze(1)
        temperature_output = normalized_output_tensor**(1/temperature)
        return Categorical(temperature_output).sample()


class MultiHexGame():
    '''
    takes a list of HexBoards as input and playes them with a list of either one or two models
    play_moves controls batched_single_move and returns the tensor triple if there is no game left to play
    batched_single_move makes one move in each of the playable games and returns the evaluation of the games or nothing if there is no game to play
    noise_parameters controls the spread of the noise
    temperature controls move selection from the predictions from 0 (take best prediction) to large positive number (take any move)
    '''
    def __init__(self, boards, models, device, noise, noise_parameters, temperature):
        self.boards = boards
        self.board_size = self.boards[0].size
        self.batch_size = len(boards)
        self.models = [nn.DataParallel(model).to(device) for model in models]
        self.noise = noise
        self.noise_parameters = noise_parameters
        self.temperature = temperature
        self.output_boards_tensor = torch.Tensor(device='cpu')
        self.positions_tensor = torch.LongTensor(device='cpu')
        self.targets_tensor = None
        self.targets_list = [[] for idx in range(self.batch_size)]
        self.reverse_winner = 1
        self.device = device

    def __repr__(self):
        return ''.join([str(board) for board in self.boards])

    def play_moves(self):
        while True:
            for model in self.models:
                self.batched_single_move(model)
                if self.current_boards == []:
                    self.positions_tensor = self.positions_tensor.view(-1, 1)
                    self.targets_tensor = torch.tensor(zip_list_of_lists_first_dim_reversed(*self.targets_list), dtype=torch.float, device=torch.device('cpu'))
                    return self.output_boards_tensor, self.positions_tensor, self.targets_tensor
                self.reverse_winner = 1 - self.reverse_winner


    def batched_single_move(self, model):        
        self.current_boards = []
        self.current_boards_tensor = torch.Tensor()
        for board_idx in range(self.batch_size):
            if self.boards[board_idx].winner == False:
                self.current_boards.append(board_idx)
                self.current_boards_tensor = torch.cat((self.current_boards_tensor, self.boards[board_idx].board_tensor.unsqueeze(0)))

        if self.current_boards == []:
            return
        
        self.current_boards_tensor = self.current_boards_tensor.to(self.device)

        with torch.no_grad():
            outputs_tensor = model(self.current_boards_tensor).detach()

        if self.noise == 'singh':
            noise_alpha, noise_beta, noise_lambda = self.noise_parameters
            outputs_tensor = singh_maddala_onto_output(outputs_tensor, noise_alpha, noise_beta, noise_lambda)

        positions1d = tempered_moves_selection(outputs_tensor, self.temperature)

        self.output_boards_tensor = torch.cat((self.output_boards_tensor, self.current_boards_tensor.detach().cpu()))
        self.positions_tensor = torch.cat((self.positions_tensor, positions1d.detach().cpu()))

        for idx in range(len(self.current_boards)):
            self.boards[self.current_boards[idx]].set_stone((int(positions1d[idx] / self.board_size), int(positions1d[idx] % self.board_size)))
            self.targets_list[self.current_boards[idx]].append(self.reverse_winner)
        return outputs_tensor
