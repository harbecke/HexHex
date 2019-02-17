import torch
from torch.distributions.categorical import Categorical
from torch.distributions.dirichlet import Dirichlet
from utils import zip_list_of_lists_first_dim_reversed


def dirichlet_onto_output(output_tensor, dirichlet_level=0.25, dirichlet_alpha=0.03):
    noise = Dirichlet(torch.full_like(output_tensor, dirichlet_alpha)).sample()
    return output_tensor * torch.exp(dirichlet_level*noise)


def tempered_moves_selection(output_tensor, temperature=0.1):
    #needs batch dimension to work
    if temperature == 0:
        return output_tensor.argmax(1)
    else:
        normalized_output_tensor = output_tensor/output_tensor.max(1)[0].unsqueeze(1)
        temperature_output = normalized_output_tensor**(1/temperature)
        return Categorical(temperature_output).sample()


class MultiHexGame():
    def __init__(self, boards, models, device, noise_level=0, noise_alpha=0.03, temperature=1):
        self.boards = boards
        self.board_size = self.boards[0].size
        self.batch_size = len(boards)
        self.models = [model.to(device) for model in models]
        self.noise_level = noise_level
        self.noise_alpha = noise_alpha
        self.temperature = temperature
        self.output_boards_tensor = torch.Tensor(device='cpu')
        self.positions_tensor = torch.LongTensor(device='cpu')
        self.targets_tensor = None
        self.targets_list = [[] for idx in range(self.batch_size)]
        self.moves_count = 0
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
                self.moves_count += 1


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

        outputs_with_dirichlet = dirichlet_onto_output(outputs_tensor, self.noise_level, self.noise_alpha)
        positions1d = tempered_moves_selection(outputs_with_dirichlet, self.temperature)

        self.output_boards_tensor = torch.cat((self.output_boards_tensor, self.current_boards_tensor.detach().cpu()))
        self.positions_tensor = torch.cat((self.positions_tensor, positions1d.detach().cpu()))

        for idx in range(len(self.current_boards)):
            self.boards[self.current_boards[idx]].set_stone((int(positions1d[idx] / self.board_size), int(positions1d[idx] % self.board_size)))
            self.targets_list[self.current_boards[idx]].append(1-self.moves_count%2)
        return outputs_tensor
