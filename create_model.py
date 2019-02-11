import torch
from hexconvolution import NoMCTSModel

model = NoMCTSModel(board_size=11, layers=5, intermediate_channels=256)
torch.save(model, 'models/first_test.pt')