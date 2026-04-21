import { NUM_CELLS } from "./constants";
import { fullNeighbors, VirtualNode } from "./coords";

export type Player = "0" | "1"; // "0" = red (top↔bottom), "1" = blue (left↔right)
export type Cell = Player | null;

const PLAYER_STARTS: Record<Player, VirtualNode> = { "0": "red-top", "1": "blue-left" };
const PLAYER_TARGETS: Record<Player, VirtualNode> = { "0": "red-bottom", "1": "blue-right" };

export function hasWinner(cells: Cell[]): boolean {
  for (const player of ["0", "1"] as Player[]) {
    const start = PLAYER_STARTS[player];
    const target = PLAYER_TARGETS[player];
    const visited = new Set<VirtualNode>();
    const stack: VirtualNode[] = [start];

    while (stack.length > 0) {
      const node = stack.pop()!;
      for (const nb of fullNeighbors(node)) {
        if (nb === target) return true;
        if (typeof nb === "string") continue;
        if (cells[nb] === player && !visited.has(nb)) {
          visited.add(nb);
          stack.push(nb);
        }
      }
    }
  }
  return false;
}

/**
 * Returns the set of cell ids forming a winning path for `player`, or null if none exists.
 * Uses BFS with parent tracking to reconstruct a shortest path.
 */
export function findWinningPath(cells: Cell[], player: Player): Set<number> | null {
  const start = PLAYER_STARTS[player];
  const target = PLAYER_TARGETS[player];
  // parent maps each visited node to the node it was reached from; start maps to itself as sentinel
  const parent = new Map<VirtualNode, VirtualNode>();
  parent.set(start, start);
  const queue: VirtualNode[] = [start];

  let found = false;
  outer: while (queue.length > 0) {
    const node = queue.shift()!;
    for (const nb of fullNeighbors(node)) {
      if (parent.has(nb)) continue;
      if (nb === target) {
        parent.set(nb, node);
        found = true;
        break outer;
      }
      if (typeof nb === "string") continue;
      if (cells[nb] === player) {
        parent.set(nb, node);
        queue.push(nb);
      }
    }
  }

  if (!found) return null;

  const path = new Set<number>();
  let cur: VirtualNode = target;
  while (cur !== start) {
    const prev = parent.get(cur)!;
    if (typeof cur === "number") path.add(cur);
    cur = prev;
  }
  return path;
}

/**
 * Minimax endgame solver.
 * Returns [score, bestMove]:  +1 = red wins, -1 = blue wins, 0 = unknown at this depth.
 */
export function minimax(
  board: Cell[],
  depth: number,
  currentPlayer: Player,
  firstPlayer: Player
): [number, number | null] {
  const other: Player = currentPlayer === "0" ? "1" : "0";

  if (hasWinner(board)) {
    return [currentPlayer === "0" ? -1 : 1, null]; // other player just won
  }
  if (depth === 0) return [0, null];

  if (currentPlayer === "0") {
    // Red: maximising
    let value = -10;
    let best: number | null = null;
    for (let i = 0; i < NUM_CELLS; i++) {
      if (board[i] !== null) continue;
      board[i] = currentPlayer;
      const [v] = minimax(board, depth - 1, other, firstPlayer);
      board[i] = null;
      if (v > value) { value = v; best = i; }
      if (value >= 1) return [value, best];
      if (currentPlayer !== firstPlayer && value === 0) return [value, best];
    }
    return [value, best];
  } else {
    // Blue: minimising
    let value = 10;
    let best: number | null = null;
    for (let i = 0; i < NUM_CELLS; i++) {
      if (board[i] !== null) continue;
      board[i] = currentPlayer;
      const [v] = minimax(board, depth - 1, other, firstPlayer);
      board[i] = null;
      if (v < value) {
        value = v; best = i;
        if (value <= -1) return [value, best];
        if (currentPlayer !== firstPlayer && value === 0) return [value, best];
      }
    }
    return [value, best];
  }
}

/** Searches depth 1 then 3 for a forced win. Returns cell id or null. */
export function findSureWinMove(board: Cell[], player: Player): number | null {
  for (const depth of [1, 3]) {
    const [score, move] = minimax(board, depth, player, player);
    if (player === "0" && score > 0) return move;
    if (player === "1" && score < 0) return move;
  }
  return null;
}
