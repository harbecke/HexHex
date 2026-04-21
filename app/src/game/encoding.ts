import { BOARD_SIZE } from "./constants";
import { posToId } from "./coords";
import { Cell } from "./rules";

/**
 * Encode the board into two Float32Array inputs for ONNX inference.
 *
 * The model always runs from the perspective of the agent (who plays the role
 * of "red" internally). The board is padded by 1 cell on all sides (13×13).
 * Border cells in channel 0 are set to 1 along the agent's own borders,
 * and channel 1 marks the opponent's borders.
 *
 * When agentIsBlue the loop runs x-major (outer=x, inner=y), which effectively
 * transposes the board so the blue agent's left↔right connectivity maps to
 * the model's expected top↔bottom direction.
 *
 * input2 is a 180° rotation of input1 (element-wise reversal per channel),
 * used to average two forward passes for rotation invariance.
 */
export function encodeBoard(
  cells: Cell[],
  agentIsBlue: boolean,
  boardSize: number = BOARD_SIZE
): { input1: Float32Array; input2: Float32Array } {
  const N = boardSize + 2;
  const channelSize = N * N;
  const input1 = new Float32Array(2 * channelSize);

  // When agentIsBlue:  channel0 = agent's stones (blue="1"), bc0=1 → left/right borders lit
  // When !agentIsBlue: channel0 = agent's stones (red="0"),  bc0=0 → top/bottom borders lit
  const agentToken = agentIsBlue ? "1" : "0";
  const opponentToken = agentIsBlue ? "0" : "1";
  const bc0 = agentIsBlue ? 1 : 0;
  const bc1 = agentIsBlue ? 0 : 1;

  function cellValue(x: number, y: number, bc: number, token: string): number {
    const onX = x === -1 || x === boardSize;
    const onY = y === -1 || y === boardSize;
    if (onX && onY) return 0; // corner: neither player
    if (onX) return bc ? 1 : 0; // left/right = blue border
    if (onY) return bc ? 0 : 1; // top/bottom = red border
    return cells[posToId(x, y)] === token ? 1 : 0;
  }

  let idx = 0;
  for (let a = -1; a <= boardSize; a++) {
    for (let b = -1; b <= boardSize; b++) {
      const [x, y] = agentIsBlue ? [a, b] : [b, a];
      input1[idx++] = cellValue(x, y, bc0, agentToken);
    }
  }
  for (let a = -1; a <= boardSize; a++) {
    for (let b = -1; b <= boardSize; b++) {
      const [x, y] = agentIsBlue ? [a, b] : [b, a];
      input1[idx++] = cellValue(x, y, bc1, opponentToken);
    }
  }

  // input2: 180° rotation = element-wise reversal per channel
  const input2 = new Float32Array(2 * channelSize);
  for (let i = 0; i < channelSize; i++) {
    input2[i] = input1[channelSize - 1 - i];
    input2[channelSize + i] = input1[2 * channelSize - 1 - i];
  }

  return { input1, input2 };
}

/**
 * Average two model outputs (original + rotated board) for rotation invariance,
 * then re-index scores into standard posToId order.
 *
 * When agentIsBlue, the encoding was x-major so model output element k corresponds
 * to board cell (k÷boardSize, k%boardSize). We must transpose back to posToId order.
 */
export function averageOutputs(
  out1: Float32Array,
  out2: Float32Array,
  agentIsBlue: boolean,
  boardSize: number = BOARD_SIZE
): Float32Array {
  const n = boardSize * boardSize;

  // Average with rotation: out2 is rotated 180°, so its element (n-1-i) corresponds to cell i
  const avg = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    avg[i] = (out1[i] + out2[n - 1 - i]) / 2;
  }

  if (!agentIsBlue) return avg;

  // Blue case: avg[k] is score for cell at (k÷boardSize, k%boardSize) in x-major order.
  // Convert to standard posToId = x + boardSize*y:
  //   standard cell id i = x + boardSize*y  →  x = i%boardSize, y = i÷boardSize
  //   x-major index for same cell = x*boardSize + y
  const scores = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = i % boardSize;
    const y = Math.floor(i / boardSize);
    scores[i] = avg[x * boardSize + y];
  }
  return scores;
}
