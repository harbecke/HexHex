import torch
from torch.distributions.categorical import Categorical
from torch.distributions.dirichlet import Dirichlet


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


class HexGame():
    '''
    change naming of board & moves_tensor to fit with rest of code
    '''
    def __init__(self, board, model, device, noise_level=0, noise_alpha=0.03, temperature=1):
        self.board = board
        self.model = model.to(device)
        self.noise_level = noise_level
        self.noise_alpha = noise_alpha
        self.temperature = temperature
        self.moves_tensor = torch.Tensor(device='cpu')
        self.position_tensor = torch.LongTensor(device='cpu')
        self.moves_count = 0
        self.player = 0
        self.device = device
        self.target = None

    def __repr__(self):
        return(str(self.board))

    def play_moves(self):
        while True:
            self.play_single_move()
            if self.board.winner:
                return self.moves_tensor, self.position_tensor, self.target

    def set_stone(self, pos):
        self.board.set_stone(self.player, pos)
        if not self.board.switch:
            self.player = 1 - self.player

    def play_single_move(self):
        board_tensor = self.board.board_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_tensor = self.model(board_tensor).detach()

        dirichlet_output = dirichlet_onto_output(output_tensor, self.noise_level, self.noise_alpha)

        position1d = tempered_moves_selection(dirichlet_output, self.temperature)

        self.position_tensor = torch.cat((self.position_tensor, position1d.unsqueeze(0)))
        board_tensor = board_tensor.detach().cpu()
        self.moves_tensor = torch.cat((self.moves_tensor, board_tensor))
        self.moves_count += 1

        self.set_stone((int(position1d / self.board.size), int(position1d % self.board.size)))

        if self.board.winner:
            self.target = torch.tensor([1.] * (self.moves_count % 2) + [0., 1.] * int(self.moves_count / 2))
        return output_tensor


class HexGameTwoModels():
    def __init__(self, board, model1, model2, device, temperature):
        self.board = board
        self.models = (model1.to(device), model2.to(device))
        #self.moves_tensor = torch.Tensor(device='cpu')
        #self.position_tensor = torch.LongTensor(device='cpu')
        #self.moves_count = 0
        self.temperature = temperature
        self.player = 0
        self.device = device

    def __repr__(self):
        return(str(self.board))

    def play_moves(self):
        while True:
            for idx in range(2):
                board_tensor = self.board.board_tensor.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output_tensor = self.models[idx](board_tensor).detach()

                position1d = tempered_moves_selection(output_tensor, self.temperature)

                position2d = (int(position1d/self.board.size), int(position1d%self.board.size))

                #self.position_tensor = torch.cat((self.position_tensor, torch.tensor(position2d).unsqueeze(0)))
                #board_tensor = board_tensor.detach().cpu()
                #self.moves_tensor = torch.cat((self.moves_tensor, board_tensor))
                #self.moves_count += 1

                self.board.set_stone(self.player, position2d)
                if self.board.switch==False:
                    self.player = 1-self.player

                if self.board.winner:
                    return idx
