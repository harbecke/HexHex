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


class SkipLayerAlpha(nn.Module):

    def __init__(self, channels, reach):
        super(SkipLayerAlpha, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=reach*2+1, padding=reach)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=reach*2+1, padding=reach)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        x = x + self.bn2(self.conv2(y))
        return F.relu(x)


class SkipLayerStar(nn.Module):

    def __init__(self, board_size, channels, reach):
        super(SkipLayerStar, self).__init__()
        self.board_size = board_size
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=reach*2+1, padding=reach)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=reach*2+1, padding=reach)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.fc1 = nn.Linear(board_size**2 * channels, channels)
        self.fc2 = nn.Linear(board_size**2 * channels, channels)

    def forward(self, x):
        y = swish(self.bn1(x))
        y = self.conv1(y) + self.fc1(torch.flatten(y, start_dim=1)).view(-1, self.channels, 1, 1).expand(-1, -1, self.board_size, self.board_size)
        y = swish(self.bn2(x))
        y = self.conv2(y) + self.fc2(torch.flatten(y, start_dim=1)).view(-1, self.channels, 1, 1).expand(-1, -1, self.board_size, self.board_size)
        return x + y


class ActivNormConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=1, padding=0):
        super(ActivNormConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=1, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class InceptionLayer(nn.Module):
    def __init__(self, channels, reach, scale=1.0):
        super(InceptionLayer, self).__init__()
        self.scale = scale

        self.anconv11 = ActivNormConv2d(64*channels, 6*channels)

        self.anconv21 = ActivNormConv2d(64*channels, 6*channels)
        self.anconv22 = ActivNormConv2d(6*channels, 7*channels, kernel_size=(1, reach*2+1),
                                        padding=(0, reach))
        self.anconv23 = ActivNormConv2d(7*channels, 8*channels, kernel_size=(reach*2+1, 1),
                                        padding=(reach, 0))

        self.conv3 = nn.Conv2d(14*channels, 64*channels, kernel_size=1)

    def forward(self, x):
        x1 = self.anconv11(x)
        x2 = self.anconv23(self.anconv22(self.anconv21(x)))
        out = torch.cat((x1, x2), 1)
        out = self.conv3(out)
        out = out * self.scale + x
        return F.relu(out)


class NoMCTSModel(nn.Module):
    '''
    model consists of a convolutional layer to change the number of channels from (three) input channels to intermediate channels
    then a specified amount of residual or skip-layers https://en.wikipedia.org/wiki/Residual_neural_network
    then policy_channels sum over the different channels and a fully connected layer to get output in shape of the board
    value range is (-inf, inf) 
    for training the sigmoid is taken, interpretable as probability to win the game when making this move
    for data generation and evaluation the softmax is taken to select a move
    '''
    def __init__(self, board_size, layers, intermediate_channels=256, policy_channels=2, reach_conv=1, skip_layer='single'):
        super(NoMCTSModel, self).__init__()
        self.board_size = board_size
        self.policy_channels = policy_channels
        self.conv = nn.Conv2d(2, intermediate_channels, kernel_size=reach_conv*2+1, padding=reach_conv)
        if skip_layer=='alpha':
            self.skiplayers = nn.ModuleList([SkipLayerAlpha(intermediate_channels, 1) for idx in range(layers)])
        elif skip_layer=='star':
            self.skiplayers = nn.ModuleList([SkipLayerStar(board_size, intermediate_channels, 1) for idx in range(layers)])
        else:
            self.skiplayers = nn.ModuleList([SkipLayer(intermediate_channels, 1) for idx in range(layers)])
        self.policyconv = nn.Conv2d(intermediate_channels, policy_channels, kernel_size=1)
        self.policybn = nn.BatchNorm2d(policy_channels)
        self.policylin = nn.Linear(board_size**2 * policy_channels, board_size**2)

    def forward(self, x):
        #illegal moves are given a huge negative bias, so they are never selected for play
        x_sum = torch.sum(x, dim=1).view(-1,self.board_size**2)
        illegal = x_sum * torch.exp(torch.tanh((x_sum.sum(dim=1)-1)*1000)*10).unsqueeze(1).expand_as(x_sum) - x_sum
        x = self.conv(x)
        for skiplayer in self.skiplayers:
            x = skiplayer(x)
        p = swish(self.policybn(self.policyconv(x))).view(-1, self.board_size**2 * self.policy_channels)
        return self.policylin(p) - illegal


class InceptionModel(nn.Module):
    '''
    model consists of a convolutional layer to change the number of channels from (three) input channels to intermediate channels
    then a specified amount of Inception-ResNet v2 layers
    the intermediate_channels parameter get multiplied by 64!
    then policy_channels sum over the different channels and a fully connected layer to get output in shape of the board
    '''
    def __init__(self, board_size, layers, intermediate_channels, reach=1):
        super(InceptionModel, self).__init__()
        self.board_size = board_size
        self.penultimate_channels = int(8*board_size*intermediate_channels**0.5)
        self.conv = nn.Conv2d(2, 64*intermediate_channels, kernel_size=1)
        self.inceptionlayers = nn.ModuleList([InceptionLayer(channels=intermediate_channels,
                                                             reach=reach,
                                                             scale=0.2) for idx in range(layers)])
        self.policyconv = ActivNormConv2d(64*intermediate_channels, self.penultimate_channels)
        self.policypool = nn.AvgPool2d(board_size, count_include_pad=False)
        self.policylin = nn.Linear(self.penultimate_channels, board_size**2)

    def forward(self, x):
        #illegal moves are given a huge negative bias, so they are never selected for play
        x_sum = torch.sum(x, dim=1).view(-1,self.board_size**2)
        illegal = x_sum * torch.exp(torch.tanh((x_sum.sum(dim=1)-1)*1000)*10).unsqueeze(1).expand_as(x_sum) - x_sum
        x = self.conv(x)
        for inceptionlayer in self.inceptionlayers:
            x = inceptionlayer(x)
        p = self.policypool(self.policyconv(x)).view(-1, self.penultimate_channels)
        return self.policylin(F.dropout(p, 0.2)) - illegal


class RandomModel(nn.Module):
    '''
    outputs negative values for every illegal move, 0 otherwise
    only makes completely random moves if temperature*temperature_decay > 0
    '''
    def __init__(self, board_size):
        super(RandomModel, self).__init__()
        self.board_size = board_size

    def forward(self, x):
        x_sum = torch.sum(x, dim=1).view(-1,self.board_size**2)
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
        illegal = 1000*torch.sum(x, dim=1).view(-1,self.board_size**2)
        return self.internal_model(x)-illegal


class RotationWrapperModel(nn.Module):
    '''
    evaluates input and its 180° rotation with parent model
    averages both predictions
    '''
    def __init__(self, model):
        super(RotationWrapperModel, self).__init__()
        self.board_size = model.board_size
        self.internal_model = model

    def forward(self, x):
        x_flip = torch.flip(x, [2, 3])
        y_flip = self.internal_model(x_flip)
        y = torch.flip(y_flip, [1])
        return (self.internal_model(x) + y)/2
