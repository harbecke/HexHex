import { describe, it, expect, vi, afterEach } from "vitest";
import { selectMove } from "../selectMove";
import { Cell } from "../rules";
import { BOARD_SIZE, NUM_CELLS } from "../constants";
import { posToId } from "../coords";

function emptyBoard(): Cell[] {
  return Array<Cell>(NUM_CELLS).fill(null);
}

function scoresWithPeak(peakId: number, peakValue = 5, bg = 0): Float32Array {
  const s = new Float32Array(NUM_CELLS);
  s.fill(bg);
  s[peakId] = peakValue;
  return s;
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("selectMove", () => {
  it("picks argmax at temperature 0", () => {
    const cells = emptyBoard();
    const scores = scoresWithPeak(42, 5);
    const move = selectMove(cells, scores, "0", { temperature: 0, canSwap: false });
    expect(move).toBe(42);
  });

  it("filters occupied cells when canSwap is false", () => {
    const cells = emptyBoard();
    cells[10] = "0";
    const scores = scoresWithPeak(10, 100); // highest-scoring cell is occupied
    const move = selectMove(cells, scores, "1", { temperature: 0, canSwap: false });
    expect(move).not.toBe(10);
  });

  it("allows the occupied cell when canSwap is true (swap option)", () => {
    const cells = emptyBoard();
    cells[10] = "0";
    const scores = scoresWithPeak(10, 100);
    const move = selectMove(cells, scores, "1", { temperature: 0, canSwap: true });
    expect(move).toBe(10);
  });

  it("samples stochastically at positive temperature", () => {
    const cells = emptyBoard();
    const scores = new Float32Array(NUM_CELLS);
    // A few peaked cells, the rest zero; at temperature 1 the full-board
    // softmax should spread picks across many cells, not just the peaks.
    scores[0] = 1;
    scores[1] = 1;
    scores[2] = 1;
    scores[3] = 1;
    scores[4] = 1;

    const seen = new Set<number>();
    for (let trial = 0; trial < 60; trial++) {
      const move = selectMove(cells, scores, "0", { temperature: 1, canSwap: false });
      seen.add(move);
    }
    expect(seen.size).toBeGreaterThanOrEqual(5);
  });

  it("blocks opponent's forced win instead of picking argmax", () => {
    // Build a board where blue (opponent) is one move from winning left↔right.
    // Red is picking. Red's "best" score is somewhere irrelevant; the blocker
    // should override it.
    const cells: Cell[] = emptyBoard();
    for (let x = 0; x < BOARD_SIZE - 1; x++) {
      cells[posToId(x, 0)] = "1";
    }
    const blockingCell = posToId(BOARD_SIZE - 1, 0);

    const scores = scoresWithPeak(60, 100); // irrelevant peak
    const move = selectMove(cells, scores, "0", { temperature: 0, canSwap: false });
    expect(move).toBe(blockingCell);
  });
});
