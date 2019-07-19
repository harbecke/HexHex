import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from hex.creation.noise import singh_maddala_onto_output, uniform_noise_onto_output
from hex.utils.utils import device, zip_list_of_lists_first_dim_reversed


def tempered_moves_selection(output_tensor, temperature):
    #samples with softmax from unnormalized values (if temp>0) and selects move
    if temperature < 10**(-10):
        return output_tensor.argmax(1)
    else:
        temperature_output = output_tensor/temperature
        return Categorical(logits=temperature_output).sample()


class MultiHexGame():
    '''
    takes a list of HexBoards as input and playes them with a list of either one or two models
    play_moves controls batched_single_move and returns the tensor triple if there is no game left to play
    batched_single_move makes one move in each of the playable games and returns the elo of the games or nothing if there is no game to play
    noise can be added after elo to boost random moves, noise and noise_parameters control the type of noise
    temperature controls move selection from the predictions from 0 (take best prediction) to large positive number (take any move)
    temperature_decay decays the temperature over time as a power function with base:temperature_decay and exponent:number of moves made
    '''
    def __init__(self, boards, models, noise, noise_parameters, temperature, temperature_decay):
        self.boards = boards
        self.board_size = self.boards[0].size
        self.batch_size = len(boards)
        self.models = [nn.DataParallel(model).to(device).eval() for model in models]
        self.noise = noise
        self.noise_parameters = noise_parameters
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.output_boards_tensor = torch.Tensor(device='cpu')
        self.positions_tensor = torch.LongTensor(device='cpu')
        self.targets_tensor = None
        self.targets_list = [[] for idx in range(self.batch_size)]
        self.reverse_winner = 1

    def __repr__(self):
        return ''.join([str(board) for board in self.boards])

    def play_moves(self):
        while True:
            for model in self.models:
                self.batched_single_move(model)
                if self.current_boards == []:
                    self.positions_tensor = self.positions_tensor.view(-1, 1)
                    self.targets_tensor = torch.tensor(zip_list_of_lists_first_dim_reversed(*self.targets_list), 
                        dtype=torch.float, device=torch.device('cpu'))
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
        
        self.current_boards_tensor = self.current_boards_tensor.to(device)

        with torch.no_grad():
            outputs_tensor = model(self.current_boards_tensor)

        if self.noise == 'singh':
            noise_alpha, noise_beta, noise_lambda = self.noise_parameters
            outputs_tensor = singh_maddala_onto_output(outputs_tensor, noise_alpha, noise_beta, noise_lambda)
        if self.noise == 'uniform':
            noise_probability, = self.noise_parameters
            outputs_tensor = uniform_noise_onto_output(outputs_tensor, noise_probability)

        moves_count = len(self.boards[self.current_boards[0]].made_moves)
        positions1d = tempered_moves_selection(outputs_tensor, self.temperature*self.temperature_decay**moves_count)

        self.output_boards_tensor = torch.cat((self.output_boards_tensor, self.current_boards_tensor.detach().cpu()))
        self.positions_tensor = torch.cat((self.positions_tensor, positions1d.detach().cpu()))

        for idx in range(len(self.current_boards)):
            self.boards[self.current_boards[idx]].set_stone(positions1d[idx].item())
            self.targets_list[self.current_boards[idx]].append(self.reverse_winner)
        return outputs_tensor
