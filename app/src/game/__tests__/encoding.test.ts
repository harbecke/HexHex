import { describe, it, expect } from "vitest";
import { encodeBoard, averageOutputs } from "../encoding";
import { posToId } from "../coords";
import { Cell } from "../rules";
import { BOARD_SIZE, NUM_CELLS } from "../constants";

const N = BOARD_SIZE + 2;
const CHANNEL_SIZE = N * N;

function emptyBoard(): Cell[] {
  return Array<Cell>(NUM_CELLS).fill(null);
}

describe("encodeBoard", () => {
  it("produces the correct total length", () => {
    const { input1, input2 } = encodeBoard(emptyBoard(), true);
    expect(input1.length).toBe(2 * CHANNEL_SIZE);
    expect(input2.length).toBe(2 * CHANNEL_SIZE);
  });

  it("input2 is the per-channel reversal of input1", () => {
    const board = emptyBoard();
    board[posToId(3, 7)] = "1";
    board[posToId(5, 2)] = "0";
    const { input1, input2 } = encodeBoard(board, true);
    for (let i = 0; i < CHANNEL_SIZE; i++) {
      expect(input2[i]).toBe(input1[CHANNEL_SIZE - 1 - i]);
      expect(input2[CHANNEL_SIZE + i]).toBe(input1[2 * CHANNEL_SIZE - 1 - i]);
    }
  });

  describe("blue agent (agentIsBlue=true)", () => {
    it("blue stone at (5,5) appears in channel 0", () => {
      const board = emptyBoard();
      board[posToId(5, 5)] = "1";
      const { input1 } = encodeBoard(board, true);
      // x-major encoding: position in padded grid = (x+1)*N + (y+1)
      const paddedIdx = (5 + 1) * N + (5 + 1);
      expect(input1[paddedIdx]).toBe(1);
      expect(input1[CHANNEL_SIZE + paddedIdx]).toBe(0); // channel 1 should be 0
    });

    it("red stone at (3,7) appears in channel 1", () => {
      const board = emptyBoard();
      board[posToId(3, 7)] = "0";
      const { input1 } = encodeBoard(board, true);
      const paddedIdx = (3 + 1) * N + (7 + 1);
      expect(input1[paddedIdx]).toBe(0); // channel 0 (blue) = 0
      expect(input1[CHANNEL_SIZE + paddedIdx]).toBe(1); // channel 1 (red) = 1
    });

    it("left/right border cells are 1 in channel 0 (blue borders)", () => {
      const { input1 } = encodeBoard(emptyBoard(), true);
      // Left border: x=-1, y=0..10 → a=-1, b=1..11 → idx = 0*N + 1..11
      for (let b = 1; b <= BOARD_SIZE; b++) {
        expect(input1[b]).toBe(1); // channel 0, left border
        expect(input1[CHANNEL_SIZE + b]).toBe(0); // channel 1, left border
      }
    });

    it("top/bottom border cells are 1 in channel 1 (red borders)", () => {
      const { input1 } = encodeBoard(emptyBoard(), true);
      // Top border: y=-1, x=0..10 → a=0..10, b=-1 → idx = (a+1)*N + 0
      for (let a = 0; a < BOARD_SIZE; a++) {
        const idx = (a + 1) * N + 0;
        expect(input1[idx]).toBe(0); // channel 0, top border = 0
        expect(input1[CHANNEL_SIZE + idx]).toBe(1); // channel 1, top border = 1
      }
    });

    it("corner cells are 0 in both channels", () => {
      const { input1 } = encodeBoard(emptyBoard(), true);
      const corners = [0, N - 1, N * (N - 1), N * N - 1];
      for (const c of corners) {
        expect(input1[c]).toBe(0);
        expect(input1[CHANNEL_SIZE + c]).toBe(0);
      }
    });

    it("empty interior cells are 0 in both channels", () => {
      const { input1 } = encodeBoard(emptyBoard(), true);
      const paddedIdx = (5 + 1) * N + (5 + 1);
      expect(input1[paddedIdx]).toBe(0);
      expect(input1[CHANNEL_SIZE + paddedIdx]).toBe(0);
    });
  });

  describe("red agent (agentIsBlue=false)", () => {
    it("red stone at (3,7) appears in channel 0", () => {
      const board = emptyBoard();
      board[posToId(3, 7)] = "0";
      const { input1 } = encodeBoard(board, false);
      // y-major encoding: a=y, b=x → paddedIdx = (y+1)*N + (x+1)
      const paddedIdx = (7 + 1) * N + (3 + 1);
      expect(input1[paddedIdx]).toBe(1);
      expect(input1[CHANNEL_SIZE + paddedIdx]).toBe(0);
    });

    it("top/bottom border cells are 1 in channel 0 (red borders)", () => {
      const { input1 } = encodeBoard(emptyBoard(), false);
      // y-major: a=y, b=x. Top border: y=-1, a=-1 → first row: idx = 0..N-1
      // But b goes -1..BOARD_SIZE, so border at b=-1 (x=-1) within row a=0..10
      // Actually top border is a=-1 (y=-1): idx = 0*N + 0..N-1
      for (let b = 1; b <= BOARD_SIZE; b++) {
        expect(input1[b]).toBe(1); // channel 0, top border (red) = 1
      }
    });
  });
});

describe("averageOutputs", () => {
  it("averages with rotation for agentIsBlue=false", () => {
    const n = NUM_CELLS;
    const out1 = new Float32Array(n).map((_, i) => i);
    const out2 = new Float32Array(n).map((_, i) => i * 2);
    const result = averageOutputs(out1, out2, false);
    for (let i = 0; i < n; i++) {
      expect(result[i]).toBeCloseTo((out1[i] + out2[n - 1 - i]) / 2);
    }
  });

  it("transposes back to posToId order for agentIsBlue=true", () => {
    const n = NUM_CELLS;
    // Set avg[k] = k (distinct values so we can track transposition)
    // avg would be: (out1[i] + out2[n-1-i]) / 2
    // We want avg[k] = k, so out2 must cancel: set out1[i]=2*i, out2[i]=0 → avg[i]=i
    const out1 = new Float32Array(Array.from({ length: n }, (_, i) => 2 * i));
    const out2 = new Float32Array(n); // zeros
    const scores = averageOutputs(out1, out2, true);
    // scores[i] = avg[x*BOARD_SIZE + y] where x = i%BOARD_SIZE, y = floor(i/BOARD_SIZE)
    for (let i = 0; i < n; i++) {
      const x = i % BOARD_SIZE;
      const y = Math.floor(i / BOARD_SIZE);
      expect(scores[i]).toBeCloseTo(x * BOARD_SIZE + y);
    }
  });
});
