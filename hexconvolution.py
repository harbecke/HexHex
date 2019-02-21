import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x):
    return x * torch.sigmoid(x)


class SkipLayer(nn.Module):
    def __init__(self, channels, reach):
        super(SkipLayer, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=reach*2+1, padding=reach)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return swish(x + self.bn(self.conv(x)))


class NoMCTSModel(nn.Module):
    '''
    model consists of a convolutional layer to change the number of channels from (three) input channels to intermediate channels
    then a specified amount of residual or skip-layers https://en.wikipedia.org/wiki/Residual_neural_network
    then policy_channels sum over the different channels and a fully connected layer to get output in shape of the board
    the last sigmoid function converts all values to probabilities: interpretable as probability to win the game when making this move
    '''
    def __init__(self, board_size, layers, intermediate_channels=256, policy_channels=2, reach=1):
        super(NoMCTSModel, self).__init__()
        self.board_size = board_size
        self.policy_channels = policy_channels
        self.conv = nn.Conv2d(3, intermediate_channels, kernel_size=reach*2+1, padding=reach)
        self.skiplayers = nn.ModuleList([SkipLayer(intermediate_channels, reach) for idx in range(layers)])
        self.policyconv = nn.Conv2d(intermediate_channels, policy_channels, kernel_size=1)
        self.policybn = nn.BatchNorm2d(policy_channels)
        self.policylin = nn.Linear(board_size**2 * policy_channels, board_size**2)

    def forward(self, x):
        #illegal moves are given a huge negative bias, so they are never selected for play - problem with noise?
        x_sum = (x[:,0]+x[:,1]).view(-1,self.board_size**2)
        illegal = x_sum * torch.exp(torch.tanh((x_sum.sum(dim=1)-1)*1000)*10).unsqueeze(1).expand_as(x_sum) - x_sum
        x = self.conv(x)
        for skiplayer in self.skiplayers:
            x = skiplayer(x)
        p = swish(self.policybn(self.policyconv(x))).view(-1, self.board_size**2 * self.policy_channels)
        return torch.sigmoid(self.policylin(p) - illegal)
