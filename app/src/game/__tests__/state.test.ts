import { describe, expect, it } from "vitest";
import { gameReducer, initialState, canSwap, GameState } from "../state";
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

describe("human swap (pie rule)", () => {
  it("human blue can swap after AI red's first move", () => {
    let state = gameReducer(initialState(), { type: "START_GAME", redIsHuman: false, blueIsHuman: true });
    state = gameReducer(state, { type: "AI_MOVE", cellId: 5, scores: dummyScores() });
    expect(canSwap(state)).toBe(true);

    state = gameReducer(state, { type: "SWAP" });

    expect(state.redIsHuman).toBe(true);
    expect(state.blueIsHuman).toBe(false);
    expect(state.cells[5]).toBe("0"); // stone color unchanged on board
    expect(state.swapUsed).toBe(true);
    expect(state.status).toBe("thinking"); // AI now plays blue
    expect(state.agentIsBlue).toBe(true);
  });

  it("SWAP is a no-op when swap is no longer available", () => {
    let state = gameReducer(initialState(), { type: "START_GAME", redIsHuman: false, blueIsHuman: true });
    state = gameReducer(state, { type: "AI_MOVE", cellId: 5, scores: dummyScores() });
    state = gameReducer(state, { type: "SWAP" });
    const afterFirstSwap = state;

    // Trying to swap again (e.g., AI returns occupied cell) should be blocked.
    state = gameReducer(state, { type: "SWAP" });
    expect(state).toBe(afterFirstSwap);
  });

  it("AI_MOVE double-swap is ignored", () => {
    let state = gameReducer(initialState(), { type: "START_GAME", redIsHuman: false, blueIsHuman: true });
    state = gameReducer(state, { type: "AI_MOVE", cellId: 5, scores: dummyScores() });
    state = gameReducer(state, { type: "SWAP" }); // human swap: red=human, blue=AI
    const afterFirstSwap = state;

    // AI tries to swap back by returning the occupied cell
    state = gameReducer(state, { type: "AI_MOVE", cellId: 5, scores: dummyScores() });
    expect(state).toBe(afterFirstSwap);
  });

  it("canSwap becomes false once the second player plays a normal move", () => {
    let state = gameReducer(initialState(), { type: "START_GAME", redIsHuman: true, blueIsHuman: true });
    state = gameReducer(state, { type: "PLAYER_MOVE", cellId: 0 });
    expect(canSwap(state)).toBe(true);
    state = gameReducer(state, { type: "PLAYER_MOVE", cellId: 1 });
    expect(canSwap(state)).toBe(false);
    expect(state.swapUsed).toBe(true);
  });
});

describe("temperature", () => {
  it("defaults to DEFAULT_TEMPERATURE", () => {
    const state = initialState();
    expect(state.temperature).toBeGreaterThan(0);
  });

  it("SET_TEMPERATURE clamps to [0, 2]", () => {
    let state = initialState();
    state = gameReducer(state, { type: "SET_TEMPERATURE", value: -1 });
    expect(state.temperature).toBe(0);
    state = gameReducer(state, { type: "SET_TEMPERATURE", value: 5 });
    expect(state.temperature).toBe(2);
    state = gameReducer(state, { type: "SET_TEMPERATURE", value: 0.7 });
    expect(state.temperature).toBeCloseTo(0.7);
  });

  it("RESET preserves temperature", () => {
    let state = initialState();
    state = gameReducer(state, { type: "SET_TEMPERATURE", value: 1.2 });
    state = gameReducer(state, { type: "START_GAME", redIsHuman: true, blueIsHuman: false });
    state = gameReducer(state, { type: "RESET" });
    expect(state.temperature).toBeCloseTo(1.2);
  });
});
