import { describe, expect, it } from "vitest";
import { classifyMoveQuality, getTopSuggestions } from "../teacher";
import { NUM_CELLS } from "../constants";

function scoresFrom(map: Record<number, number>): (number | null)[] {
  const s: (number | null)[] = Array(NUM_CELLS).fill(null);
  for (const [k, v] of Object.entries(map)) s[Number(k)] = v;
  return s;
}

function emptyCells(): (string | null)[] {
  return Array(NUM_CELLS).fill(null);
}

describe("classifyMoveQuality", () => {
  it("labels the top move as Good", () => {
    const scores = scoresFrom({ 0: 2.0, 1: 0.0, 2: -1.5 });
    const cells = emptyCells();
    cells[0] = "0"; // just played cell 0
    expect(classifyMoveQuality(scores, cells, 0)?.label).toBe("Good");
  });

  it("uses delta thresholds 0.5 / 1.5 / 3.0", () => {
    const scores = scoresFrom({ 0: 3.0, 1: 2.6, 2: 1.6, 3: 0.1, 4: -1.0 });
    const mk = (played: number) => {
      const c = emptyCells();
      c[played] = "0";
      return c;
    };
    expect(classifyMoveQuality(scores, mk(1), 1)?.label).toBe("Good"); // Δ=0.4
    expect(classifyMoveQuality(scores, mk(2), 2)?.label).toBe("Inaccuracy"); // Δ=1.4
    expect(classifyMoveQuality(scores, mk(3), 3)?.label).toBe("Mistake"); // Δ=2.9
    expect(classifyMoveQuality(scores, mk(4), 4)?.label).toBe("Blunder"); // Δ=4.0
  });

  it("returns null for missing scores", () => {
    expect(classifyMoveQuality(null, emptyCells(), 0)).toBeNull();
    expect(classifyMoveQuality(scoresFrom({}), emptyCells(), 5)).toBeNull();
  });

  it("ignores high scores on occupied cells", () => {
    // Cell 0 has a huge score but is occupied by the opponent — it's not
    // a legal move, so the played move at 1 should be the best available.
    const scores = scoresFrom({ 0: 5.0, 1: 2.0, 2: 1.5 });
    const cells = emptyCells();
    cells[0] = "1"; // opponent stone
    cells[1] = "0"; // just played
    expect(classifyMoveQuality(scores, cells, 1)?.label).toBe("Good");
    expect(classifyMoveQuality(scores, cells, 1)?.delta).toBeCloseTo(0);
  });
});

describe("getTopSuggestions", () => {
  it("only returns moves strictly better than the played move", () => {
    const scores = scoresFrom({ 0: 2.0, 1: 1.5, 2: 0.0, 3: -0.3 });
    const cells = emptyCells();
    cells[0] = "0"; // pretend 0 was just played
    const suggestions = getTopSuggestions(scores, cells, 0);
    expect(suggestions).toHaveLength(0);
  });

  it("returns up to n strictly-better empty cells, sorted by score", () => {
    const scores = scoresFrom({ 0: 2.0, 1: 0.5, 2: 1.5, 3: -0.5, 4: 1.0, 5: 1.9 });
    const cells = emptyCells();
    cells[1] = "0"; // played cell 1 (score 0.5)
    const suggestions = getTopSuggestions(scores, cells, 1, 3);
    expect(suggestions.map((s) => s.id)).toEqual([0, 5, 2]);
    expect(suggestions[0].rank).toBe(1);
    expect(suggestions[1].rank).toBe(2);
    expect(suggestions[2].rank).toBe(3);
  });

  it("skips already-occupied cells", () => {
    const scores = scoresFrom({ 0: 3.0, 1: 2.0, 2: 1.0 });
    const cells = emptyCells();
    cells[0] = "1"; // occupied by opponent
    cells[2] = "0"; // played move
    const suggestions = getTopSuggestions(scores, cells, 2, 3);
    expect(suggestions.map((s) => s.id)).toEqual([1]);
  });
});
