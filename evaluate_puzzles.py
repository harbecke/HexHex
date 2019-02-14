import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss(reduction='sum')

model = torch.load('models/first_train.pt')
model = model.to(device)
board_tensor, moves_tensor = torch.load('puzzle.pt')
board_tensor, moves_tensor = board_tensor.to(device), moves_tensor.to(device)
with torch.no_grad():
    board_tensor = board_tensor[:,:2]
    full_pred_tensor = model(board_tensor)
    local_predictions = torch.gather(full_pred_tensor, 1, moves_tensor)
    loss = criterion(local_predictions, torch.ones_like(moves_tensor, dtype=torch.float, device=device))
    print(local_predictions)#loss.item())