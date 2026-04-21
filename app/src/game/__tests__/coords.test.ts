import { describe, it, expect } from "vitest";
import { posToId, idToPos, neighbors, fullNeighbors } from "../coords";
import { BOARD_SIZE } from "../constants";

describe("posToId / idToPos", () => {
  it("top-left corner", () => {
    expect(posToId(0, 0)).toBe(0);
    expect(idToPos(0)).toEqual([0, 0]);
  });

  it("bottom-right corner", () => {
    expect(posToId(10, 10)).toBe(120);
    expect(idToPos(120)).toEqual([10, 10]);
  });

  it("round-trips for sample cells", () => {
    for (const [x, y] of [[0,0],[5,5],[3,7],[10,0],[0,10]] as [number,number][]) {
      expect(idToPos(posToId(x, y))).toEqual([x, y]);
    }
  });
});

describe("neighbors", () => {
  it("interior cell has exactly 6 neighbors", () => {
    expect(neighbors(posToId(5, 5))).toHaveLength(6);
  });

  it("corner (0,0) has exactly 2 neighbors", () => {
    expect(neighbors(posToId(0, 0))).toHaveLength(2);
  });

  it("corner (10,10) has exactly 2 neighbors", () => {
    expect(neighbors(posToId(10, 10))).toHaveLength(2);
  });

  it("left-edge cell (0,5) has exactly 4 neighbors", () => {
    // Has (0,4), (0,6), (1,4), (1,5) — the hex grid shear adds right-diagonal neighbors
    expect(neighbors(posToId(0, 5))).toHaveLength(4);
  });

  it("top-right corner (10,0) has exactly 3 neighbors", () => {
    // Has (9,0), (9,1), (10,1)
    expect(neighbors(posToId(10, 0))).toHaveLength(3);
  });

  it("all neighbors are valid cell ids", () => {
    for (const nb of neighbors(posToId(5, 5))) {
      expect(nb).toBeGreaterThanOrEqual(0);
      expect(nb).toBeLessThan(BOARD_SIZE * BOARD_SIZE);
    }
  });
});

describe("fullNeighbors", () => {
  it("left-column cell includes blue-left", () => {
    expect(fullNeighbors(posToId(0, 5))).toContain("blue-left");
  });

  it("right-column cell includes blue-right", () => {
    expect(fullNeighbors(posToId(10, 5))).toContain("blue-right");
  });

  it("top-row cell includes red-top", () => {
    expect(fullNeighbors(posToId(5, 0))).toContain("red-top");
  });

  it("bottom-row cell includes red-bottom", () => {
    expect(fullNeighbors(posToId(5, 10))).toContain("red-bottom");
  });

  it("interior cell has no virtual nodes", () => {
    const nbs = fullNeighbors(posToId(5, 5));
    expect(nbs.every((n) => typeof n === "number")).toBe(true);
  });

  it("red-top returns all 11 top-row ids", () => {
    const nbs = fullNeighbors("red-top");
    expect(nbs).toHaveLength(BOARD_SIZE);
    for (let x = 0; x < BOARD_SIZE; x++) {
      expect(nbs).toContain(posToId(x, 0));
    }
  });

  it("blue-left returns all 11 left-column ids", () => {
    const nbs = fullNeighbors("blue-left");
    expect(nbs).toHaveLength(BOARD_SIZE);
    for (let y = 0; y < BOARD_SIZE; y++) {
      expect(nbs).toContain(posToId(0, y));
    }
  });
});
