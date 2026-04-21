import { NUM_CELLS } from "./constants";
import { Cell, Player, findSureWinMove } from "./rules";

export interface SelectMoveOptions {
  temperature: number;
  canSwap: boolean;
}

/**
 * Pick an AI move from the model's raw scores.
 *
 * - Filters to legal cells (empties + the single occupied cell if `canSwap`).
 * - When `temperature > 0`, samples from a softmax over all legal cells.
 *   Otherwise picks the argmax.
 * - After selection, if placing the chosen move would leave the opponent with
 *   a forced win, plays that blocking cell instead.
 */
export function selectMove(
  cells: Cell[],
  scores: Float32Array,
  agentPlayer: Player,
  options: SelectMoveOptions
): number {
  const { temperature, canSwap } = options;

  const legal: number[] = [];
  for (let i = 0; i < NUM_CELLS; i++) {
    if (cells[i] !== null && !canSwap) continue;
    legal.push(i);
  }

  let picked: number;
  if (temperature <= 0 || legal.length <= 1) {
    picked = legal[0];
    for (let i = 1; i < legal.length; i++) {
      if (scores[legal[i]] > scores[picked]) picked = legal[i];
    }
  } else {
    let maxScaled = -Infinity;
    for (const i of legal) {
      const s = scores[i] / temperature;
      if (s > maxScaled) maxScaled = s;
    }
    const exps = legal.map((i) => Math.exp(scores[i] / temperature - maxScaled));
    const total = exps.reduce((a, b) => a + b, 0);
    const r = Math.random() * total;
    let acc = 0;
    picked = legal[legal.length - 1];
    for (let i = 0; i < legal.length; i++) {
      acc += exps[i];
      if (r <= acc) {
        picked = legal[i];
        break;
      }
    }
  }

  // If the picked move is a swap (occupied cell), play it as-is — the reducer
  // handles the swap. Otherwise, check whether it would let the opponent win.
  if (cells[picked] !== null) return picked;

  const opponent: Player = agentPlayer === "0" ? "1" : "0";
  const testCells = cells.slice();
  testCells[picked] = agentPlayer;
  const forcedBlock = findSureWinMove(testCells, opponent);
  return forcedBlock !== null ? forcedBlock : picked;
}
