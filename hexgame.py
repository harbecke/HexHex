import torch


class HexGame():
    def __init__(self, board, model):
        self.board = board
        self.model = model
        self.moves_tensor = torch.Tensor()
        self.position_tensor = torch.LongTensor()
        self.moves_count = 0
        self.player = 0

    def __repr__(self):
        return(str(self.board))

    def play_moves(self):
        while True:

            board_tensor = self.board.board_tensor.unsqueeze(0)

            with torch.no_grad():
                output_values = self.model(board_tensor)

            position1d = output_values.argmax()
            position2d = (int(position1d/self.board.size), int(position1d%self.board.size))

            self.position_tensor = torch.cat((self.position_tensor, torch.tensor(position2d).unsqueeze(0)))
            self.moves_tensor = torch.cat((self.moves_tensor, board_tensor))
            self.moves_count += 1
            
            self.board.set_stone(self.player, position2d)
            self.player = 1-self.player

            if self.board.winner:

                if self.player == 1:
                    self.target = torch.tensor([1]+[0,1]*int(self.moves_count/2))
                    
                else:
                    self.target = torch.tensor([0,1]*int(self.moves_count/2))
                
                return self.moves_tensor, self.position_tensor, self.target