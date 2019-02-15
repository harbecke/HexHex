import torch
from torch.distributions.categorical import Categorical

def model_evaluates_with_noise_temperature(board_tensor, model, noise, noise_level=0, temperature=1):
    """
    have to switch temperature with noise

    :return (chosen move, original move ratings)
    """
    with torch.no_grad():
        output_values = model(board_tensor)
        output_values = output_values.detach().cpu()

    noisy_output_value = output_values
    if noise_level > 0:
        noisy_output_value = noisy_output_value * torch.exp(noise_level*noise.sample())

    if temperature == 0:
        return noisy_output_value.argmax(), output_values
    else:
        temperature_output = torch.expm1(noisy_output_value)**(1/temperature)
        return Categorical(temperature_output).sample(), output_values


class HexGame():
    '''
    change naming of board & moves_tensor to fit with rest of code
    '''
    def __init__(self, board, model, device, noise=None, noise_level=0, temperature=1):
        self.board = board
        self.model = model.to(device)
        self.noise = noise
        self.noise_level = noise_level
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
            result = self.play_single_move()
            if self.board.winner:
                return result

    def set_stone(self, pos):
        self.board.set_stone(self.player, pos)
        if not self.board.switch:
            self.player = 1 - self.player

    def play_single_move(self):
        board_tensor = self.board.board_tensor.unsqueeze(0).to(self.device)

        position1d, move_ratings = model_evaluates_with_noise_temperature(board_tensor, self.model, self.noise,
                                                                          self.noise_level,
                                                                          self.temperature)

        self.position_tensor = torch.cat((self.position_tensor, position1d.unsqueeze(0)))
        board_tensor = board_tensor.detach().cpu()
        self.moves_tensor = torch.cat((self.moves_tensor, board_tensor))
        self.moves_count += 1

        self.set_stone((int(position1d / self.board.size), int(position1d % self.board.size)))

        if self.board.winner:
            self.target = torch.tensor([1.] * (self.moves_count % 2) + [0., 1.] * int(self.moves_count / 2))
        return self.moves_tensor, self.position_tensor, self.target, move_ratings


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
                    position1d, _ = model_evaluates_with_noise_temperature(board_tensor, self.models[idx],
                                                                                     False, 0, self.temperature)

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
