import { describe, it, expect } from "vitest";
import { hasWinner, minimax, findSureWinMove, Cell } from "../rules";
import { posToId } from "../coords";
import { BOARD_SIZE, NUM_CELLS } from "../constants";

function emptyBoard(): Cell[] {
  return Array<Cell>(NUM_CELLS).fill(null);
}

describe("hasWinner", () => {
  it("returns false for empty board", () => {
    expect(hasWinner(emptyBoard())).toBe(false);
  });

  it("detects red win (top-to-bottom path)", () => {
    const board = emptyBoard();
    for (let y = 0; y < BOARD_SIZE; y++) {
      board[posToId(0, y)] = "0"; // left column, all red
    }
    expect(hasWinner(board)).toBe(true);
  });

  it("detects blue win (left-to-right path)", () => {
    const board = emptyBoard();
    for (let x = 0; x < BOARD_SIZE; x++) {
      board[posToId(x, 0)] = "1"; // top row, all blue
    }
    expect(hasWinner(board)).toBe(true);
  });

  it("returns false for incomplete red path", () => {
    const board = emptyBoard();
    for (let y = 0; y < BOARD_SIZE - 1; y++) {
      board[posToId(0, y)] = "0"; // one gap at the bottom
    }
    expect(hasWinner(board)).toBe(false);
  });

  it("returns false for incomplete blue path", () => {
    const board = emptyBoard();
    for (let x = 1; x < BOARD_SIZE; x++) {
      board[posToId(x, 0)] = "1"; // gap at left edge
    }
    expect(hasWinner(board)).toBe(false);
  });

  it("returns false for disconnected partial paths", () => {
    const board = emptyBoard();
    board[posToId(0, 0)] = "0";
    board[posToId(5, 5)] = "0";
    board[posToId(10, 10)] = "0";
    expect(hasWinner(board)).toBe(false);
  });
});

describe("minimax", () => {
  it("returns [0, null] at depth 0", () => {
    const board = emptyBoard();
    expect(minimax(board, 0, "0", "0")).toEqual([0, null]);
    expect(minimax(board, 0, "1", "1")).toEqual([0, null]);
  });

  it("red wins in one move when only one cell left", () => {
    // Fill all cells except the one that completes red's path
    const board = emptyBoard();
    // Build a near-complete red path: x=0, y=0..9, missing y=10
    for (let y = 0; y < BOARD_SIZE - 1; y++) {
      board[posToId(0, y)] = "0";
    }
    const missingId = posToId(0, BOARD_SIZE - 1);
    const [score, move] = minimax(board, 1, "0", "0");
    expect(score).toBe(1);
    expect(move).toBe(missingId);
  });

  it("blue wins in one move when only one cell left", () => {
    const board = emptyBoard();
    for (let x = 0; x < BOARD_SIZE - 1; x++) {
      board[posToId(x, 0)] = "1";
    }
    const missingId = posToId(BOARD_SIZE - 1, 0);
    const [score, move] = minimax(board, 1, "1", "1");
    expect(score).toBe(-1);
    expect(move).toBe(missingId);
  });
});

describe("findSureWinMove", () => {
  it("returns the winning cell when red can win in 1", () => {
    const board = emptyBoard();
    for (let y = 0; y < BOARD_SIZE - 1; y++) {
      board[posToId(0, y)] = "0";
    }
    const winCell = posToId(0, BOARD_SIZE - 1);
    expect(findSureWinMove(board, "0")).toBe(winCell);
  });

  it("returns the winning cell when blue can win in 1", () => {
    const board = emptyBoard();
    for (let x = 0; x < BOARD_SIZE - 1; x++) {
      board[posToId(x, 0)] = "1";
    }
    const winCell = posToId(BOARD_SIZE - 1, 0);
    expect(findSureWinMove(board, "1")).toBe(winCell);
  });

  it("returns null on empty board", () => {
    expect(findSureWinMove(emptyBoard(), "0")).toBeNull();
    expect(findSureWinMove(emptyBoard(), "1")).toBeNull();
  });
});
