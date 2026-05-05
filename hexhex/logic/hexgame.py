import math

import torch

from hexhex.logic.temperature import select_moves
from hexhex.utils import utils


class MultiHexGame():
    """Plays multiple HexBoards in parallel using one or two models.

    `temperature_schedule` (callable: move_idx -> float) drives move selection:
    inf → uniform random (model is skipped), 0 → argmax, otherwise softmax(logits / T).

    `collect_training_data=False` disables the per-ply CPU accumulation of board
    states and chosen positions. The eval path discards them, and skipping the
    .cpu() copies removes a GPU sync each ply.
    """

    def __init__(self, boards, models, temperature_schedule, gamma=1, optimality_checker=None,
                 collect_training_data=True):
        self.boards = boards
        self.board_size = self.boards[0].size
        self.batch_size = len(boards)
        self.models = list(models)
        self.temperature_schedule = temperature_schedule
        self.collect_training_data = collect_training_data
        self.output_boards_tensor = torch.Tensor(device='cpu')
        self.positions_tensor = torch.LongTensor(device='cpu')
        self.gamma = gamma
        self.optimality_checker = optimality_checker
        self.optimal_count = 0
        self.evaluated_count = 0

    def __repr__(self):
        return ''.join([str(board) for board in self.boards])

    def play_moves(self):
        while True:
            for model in self.models:
                self.batched_single_move(model)
                if self.current_boards == []:
                    if self.collect_training_data:
                        self.positions_tensor = self.positions_tensor.view(-1, 1)
                        targets = utils.get_targets(self.boards, self.gamma)
                        return self.output_boards_tensor, self.positions_tensor, targets
                    return None, None, None

    def batched_single_move(self, model):
        self.current_boards = []
        board_tensors = []
        for board_idx in range(self.batch_size):
            if self.boards[board_idx].winner == False:
                self.current_boards.append(board_idx)
                board_tensors.append(self.boards[board_idx].board_tensor.unsqueeze(0))

        if self.current_boards == []:
            return

        self.current_boards_tensor = torch.cat(board_tensors).to(utils.device)
        moves_count = len(self.boards[self.current_boards[0]].made_moves)
        temperature = self.temperature_schedule(moves_count)

        if math.isinf(temperature):
            outputs_tensor = None
            positions1d = torch.randint(
                0, self.board_size ** 2, (len(self.current_boards),), device=utils.device)
        else:
            with torch.no_grad():
                outputs_tensor = model(self.current_boards_tensor)
            positions1d = select_moves(outputs_tensor, temperature)

        if self.collect_training_data:
            self.output_boards_tensor = torch.cat(
                (self.output_boards_tensor, self.current_boards_tensor.detach().cpu()))
            self.positions_tensor = torch.cat(
                (self.positions_tensor, positions1d.detach().cpu()))

        positions_list = positions1d.tolist()
        for idx in range(len(self.current_boards)):
            board = self.boards[self.current_boards[idx]]
            correct_position = utils.correct_position1d(positions_list[idx], self.board_size,
                board.player)
            if self.optimality_checker is not None:
                verdict = self.optimality_checker.is_optimal(board, correct_position)
                if verdict is not None:
                    self.evaluated_count += 1
                    if verdict:
                        self.optimal_count += 1
            board.set_stone(correct_position)
        return outputs_tensor
