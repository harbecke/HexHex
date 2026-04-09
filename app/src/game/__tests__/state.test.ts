import { describe, expect, it } from "vitest";
import { gameReducer, initialState, GameState } from "../state";
import { NUM_CELLS } from "../constants";

function dummyScores() {
  return new Float32Array(NUM_CELLS);
}

/** Start a standard human (red) vs AI (blue) game. */
function startedState(): GameState {
  return gameReducer(initialState(), { type: "START_GAME", redIsHuman: true, blueIsHuman: false });
}

describe("gameReducer swap behavior", () => {
  it("AI swap keeps first stone color and flips sides", () => {
    let state = startedState();
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
    let state = startedState();
    state = gameReducer(state, { type: "PLAYER_MOVE", cellId: 0 });
    state = gameReducer(state, { type: "AI_MOVE", cellId: 1, scores: dummyScores() });

    expect(state.cells[0]).toBe("0");
    expect(state.cells[1]).toBe("1");
    expect(state.agentIsBlue).toBe(true);
    expect(state.aiSwapped).toBe(false);
    expect(state.status).toBe("idle");
  });
});

describe("gameReducer player modes", () => {
  it("human vs human: both moves stay idle, no AI thinking", () => {
    let state = gameReducer(initialState(), { type: "START_GAME", redIsHuman: true, blueIsHuman: true });
    expect(state.status).toBe("idle");

    state = gameReducer(state, { type: "PLAYER_MOVE", cellId: 0 });
    expect(state.cells[0]).toBe("0");
    expect(state.status).toBe("idle");

    state = gameReducer(state, { type: "PLAYER_MOVE", cellId: 1 });
    expect(state.cells[1]).toBe("1");
    expect(state.status).toBe("idle");
  });

  it("AI vs AI: starts thinking, alternates agentIsBlue", () => {
    let state = gameReducer(initialState(), { type: "START_GAME", redIsHuman: false, blueIsHuman: false });
    expect(state.status).toBe("thinking");
    expect(state.agentIsBlue).toBe(false); // AI plays red first

    // Simulate AI (red) placing a stone
    state = gameReducer(state, { type: "AI_MOVE", cellId: 0, scores: dummyScores() });
    expect(state.cells[0]).toBe("0");
    expect(state.status).toBe("thinking");
    expect(state.agentIsBlue).toBe(true); // now AI plays blue
  });

  it("AI (red) vs human (blue): starts thinking, then idle after AI move", () => {
    let state = gameReducer(initialState(), { type: "START_GAME", redIsHuman: false, blueIsHuman: true });
    expect(state.status).toBe("thinking");
    expect(state.agentIsBlue).toBe(false);

    state = gameReducer(state, { type: "AI_MOVE", cellId: 0, scores: dummyScores() });
    expect(state.cells[0]).toBe("0");
    expect(state.status).toBe("idle");
  });
});
