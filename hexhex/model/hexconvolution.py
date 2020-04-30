import torch
import torch.nn as nn


def swish(x):
    return x * torch.sigmoid(x)


class SkipLayerBias(nn.Module):

    def __init__(self, channels, reach, scale=1):
        super(SkipLayerBias, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=reach*2+1, padding=reach, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.scale = scale

    def forward(self, x):
        return swish(x + self.scale*self.bn(self.conv(x)))


class Conv(nn.Module):
    '''
    model consists of a convolutional layer to change the number of channels from two input channels to intermediate channels
    then a specified amount of residual or skip-layers https://en.wikipedia.org/wiki/Residual_neural_network
    then policyconv reduce the intermediate channels to one
    value range is (-inf, inf) 
    for training the sigmoid is taken, interpretable as probability to win the game when making this move
    for data generation and evaluation the softmax is taken to select a move
    '''
    def __init__(self, board_size, layers, intermediate_channels, reach, export_mode):
        super(Conv, self).__init__()
        self.board_size = board_size
        self.conv = nn.Conv2d(2, intermediate_channels, kernel_size=2*reach+1, padding=reach-1)
        self.skiplayers = nn.ModuleList([SkipLayerBias(intermediate_channels, 1) for idx in range(layers)])
        self.policyconv = nn.Conv2d(intermediate_channels, 1, kernel_size=2*reach+1, padding=reach, bias=False)
        self.bias = nn.Parameter(torch.zeros(board_size**2))
        self.export_mode = export_mode

    def forward(self, x):
        x_sum = torch.sum(x[:, :, 1:-1, 1:-1], dim=1).view(-1,self.board_size**2)
        x = self.conv(x)
        for skiplayer in self.skiplayers:
            x = skiplayer(x)
        if self.export_mode:
            return self.policyconv(x).view(-1, self.board_size ** 2) + self.bias
        #  illegal moves are given a huge negative bias, so they are never selected for play
        illegal = x_sum * torch.exp(torch.tanh((x_sum.sum(dim=1)-1)*1000)*10).unsqueeze(1).expand_as(x_sum) - x_sum
        return self.policyconv(x).view(-1, self.board_size**2) + self.bias - illegal


class RandomModel(nn.Module):
    '''
    outputs negative values for every illegal move, 0 otherwise
    only makes completely random moves if temperature*temperature_decay > 0
    '''
    def __init__(self, board_size):
        super(RandomModel, self).__init__()
        self.board_size = board_size

    def forward(self, x):
        x_sum = torch.sum(x[:, :, 1:-1, 1:-1], dim=1).view(-1,self.board_size**2)
        illegal = x_sum * torch.exp(torch.tanh((x_sum.sum(dim=1)-1)*1000)*10).unsqueeze(1).expand_as(x_sum) - x_sum
        return torch.rand_like(illegal) - illegal


class NoSwitchWrapperModel(nn.Module):
    '''
    same functionality as parent model, but switching is illegal
    '''
    def __init__(self, model):
        super(NoSwitchWrapperModel, self).__init__()
        self.board_size = model.board_size
        self.internal_model = model

    def forward(self, x):
        illegal = 1000*torch.sum(x[:, :, 1:-1, 1:-1], dim=1).view(-1,self.board_size**2)
        return self.internal_model(x)-illegal


class RotationWrapperModel(nn.Module):
    '''
    evaluates input and its 180° rotation with parent model
    averages both predictions
    '''
    def __init__(self, model, export_mode):
        super(RotationWrapperModel, self).__init__()
        self.board_size = model.board_size
        self.internal_model = model
        self.export_mode = export_mode

    def forward(self, x):
        if self.export_mode:
            return self.internal_model(x)
        x_flip = torch.flip(x, [2, 3])
        y_flip = self.internal_model(x_flip)
        y = torch.flip(y_flip, [1])
        return (self.internal_model(x) + y)/2