import { describe, expect, it } from "vitest";
import { gameReducer, initialState } from "../state";
import { NUM_CELLS } from "../constants";

function dummyScores() {
  return new Float32Array(NUM_CELLS);
}

describe("gameReducer swap behavior", () => {
  it("AI swap keeps first stone color and flips sides", () => {
    let state = initialState();
    state = gameReducer(state, { type: "PLAYER_MOVE", cellId: 0 });
    expect(state.status).toBe("thinking");
    expect(state.cells[0]).toBe("0");

    state = gameReducer(state, { type: "AI_MOVE", cellId: 0, scores: dummyScores() });

    expect(state.cells[0]).toBe("0");
    expect(state.agentIsBlue).toBe(false);
    expect(state.aiSwapped).toBe(true);
    expect(state.modelScores.every((s) => s !== null)).toBe(true);
    expect(state.status).toBe("idle");
  });

  it("AI normal move still places a stone on empty cells", () => {
    let state = initialState();
    state = gameReducer(state, { type: "PLAYER_MOVE", cellId: 0 });
    state = gameReducer(state, { type: "AI_MOVE", cellId: 1, scores: dummyScores() });

    expect(state.cells[0]).toBe("0");
    expect(state.cells[1]).toBe("1");
    expect(state.agentIsBlue).toBe(true);
    expect(state.aiSwapped).toBe(false);
    expect(state.status).toBe("idle");
  });
});
