#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from hexhex.logic import temperature
from hexhex.logic.hexboard import Board
from hexhex.logic.hexgame import MultiHexGame
from hexhex.utils import utils
from hexhex.utils.logger import logger
from hexhex.utils.summary import writer


def _annotated_heatmap_figure(data_2d, fmt, cmap='YlOrRd', vmin=None, vmax=None):
    rows, cols = data_2d.shape
    vmin = np.nanmin(data_2d) if vmin is None else vmin
    vmax = np.nanmax(data_2d) if vmax is None else vmax
    fig, ax = plt.subplots(figsize=(cols * 0.9 + 0.4, rows * 0.9 + 0.4), dpi=150)
    ax.imshow(data_2d, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    for i in range(rows):
        for j in range(cols):
            val = data_2d[i, j]
            if np.isnan(val):
                text, color = '·', 'gray'
            else:
                text = format(val, fmt)
                brightness = (val - vmin) / (vmax - vmin + 1e-8)
                color = 'white' if brightness > 0.65 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=14, color=color)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout(pad=0.1)
    return fig


class SelfPlayGenerator:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.board_size = model.board_size
        self.game_lengths = []

    def self_play_game(self):
        """
        Generates data points from self play.
        yields 3 tensors containing for each move:
        - board
        - move
        - result of game for active player
        """
        boards = [Board(size=self.board_size) for _ in range(self.args.batch_size)]
        multihexgame = MultiHexGame(
            boards=boards,
            models=(self.model,),
            temperature_schedule=temperature.from_config(self.args.temperature),
            gamma=self.args.gamma,
        )
        board_states, moves, targets = multihexgame.play_moves()
        self.game_lengths.extend([len(board.made_moves) for board in boards])
        output_list = list(zip(board_states, moves, targets))
        np.random.shuffle(output_list)

        for board_state, move, target in output_list:
            yield board_state, move, target

    def position_generator(self):
        while True:
            for board_tensor, move_tensor, result_tensor in self.self_play_game():
                yield board_tensor, move_tensor, result_tensor


def create_self_play_data(args, model, num_samples, verbose=True, step=None):
    if verbose:
        logger.info("")
        logger.info("=== creating data from self play ===")

    self_play_generator = SelfPlayGenerator(model, args)
    position_generator = self_play_generator.position_generator()

    board_size = model.board_size
    all_boards_tensor = torch.zeros((num_samples, 2, board_size+2, board_size+2), dtype=torch.float)
    all_moves = torch.zeros((num_samples, 1), dtype=torch.long)
    all_results = torch.zeros(num_samples, dtype=torch.float)

    for sample_idx in range(num_samples):
        board_tensor, move, result = next(position_generator)
        all_boards_tensor[sample_idx] = board_tensor
        all_moves[sample_idx] = move
        all_results[sample_idx] = result

    if verbose:
        def k_th_move_idx(k):
            return [idx for idx in range(all_boards_tensor.shape[0]) if np.ceil(torch.sum(
                all_boards_tensor[idx, :, 1:-1, 1:-1])) == k]

        first_move_indices = k_th_move_idx(0)
        first_move_frequency = torch.zeros([board_size ** 2], dtype=torch.float)
        first_move_win_percentage = torch.zeros([board_size ** 2], dtype=torch.float)

        for x in first_move_indices:
            first_move_frequency[all_moves[x].item()] += 1
            if all_results[x].item() > 0.5:
                first_move_win_percentage[all_moves[x].item()] += 1
        first_move_win_percentage /= first_move_frequency

        with torch.no_grad():
            board = Board(model.board_size)
            logits = model(board.board_tensor.unsqueeze(0).to(utils.device)).view(board_size, board_size)
            win_probs = torch.sigmoid(logits)

        def log_grid(label, grid):
            logger.info(label)
            with np.printoptions(precision=2, suppress=True, floatmode='fixed'):
                for row in str(grid).splitlines():
                    logger.info("  " + row)

        log_grid("Self-play first move frequency:", first_move_frequency.view(board_size, board_size).int().numpy())
        log_grid("Self-play first move win rate:", first_move_win_percentage.view(board_size, board_size).numpy())
        log_grid("Model predicted first move win probabilities:", win_probs.cpu().numpy())

        avg_game_length = np.mean(self_play_generator.game_lengths)
        logger.info(f'Average game length: {avg_game_length:.1f} moves')
        writer.add_scalar('data/avg_game_length', avg_game_length, step)

        freq_fig = _annotated_heatmap_figure(
            first_move_frequency.view(board_size, board_size).numpy().astype(float), '.0f')
        writer.add_figure('data/first_move_frequency', freq_fig, step)

        win_pct_fig = _annotated_heatmap_figure(
            first_move_win_percentage.view(board_size, board_size).numpy(), '.2f',
            cmap='RdYlGn', vmin=0.0, vmax=1.0)
        writer.add_figure('data/first_move_win_rate', win_pct_fig, step)

        model_pred_fig = _annotated_heatmap_figure(
            win_probs.cpu().numpy(), '.2f',
            cmap='RdYlGn', vmin=0.0, vmax=1.0)
        writer.add_figure('data/model_first_move_prediction', model_pred_fig, step)

        logger.info(f'=== created self-play data ===')

    return [all_boards_tensor, all_moves, all_results]
