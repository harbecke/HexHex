import { describe, expect, it } from "vitest";
import { gameReducer, initialState, canSwap, DEFAULT_TEMPERATURE, GameState } from "../state";
import { NUM_CELLS } from "../constants";

function dummyScores() {
  return new Float32Array(NUM_CELLS);
}

/** Start a standard human (red) vs AI (blue) game. */
function startedState(): GameState {
  return gameReducer(initialState(), {
    type: "START_GAME",
    redIsHuman: true,
    blueIsHuman: false,
    redTemperature: DEFAULT_TEMPERATURE,
    blueTemperature: DEFAULT_TEMPERATURE,
  });
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
    let state = gameReducer(initialState(), {
      type: "START_GAME",
      redIsHuman: true,
      blueIsHuman: true,
      redTemperature: DEFAULT_TEMPERATURE,
      blueTemperature: DEFAULT_TEMPERATURE,
    });
    expect(state.status).toBe("idle");

    state = gameReducer(state, { type: "PLAYER_MOVE", cellId: 0 });
    expect(state.cells[0]).toBe("0");
    expect(state.status).toBe("idle");

    state = gameReducer(state, { type: "PLAYER_MOVE", cellId: 1 });
    expect(state.cells[1]).toBe("1");
    expect(state.status).toBe("idle");
  });

  it("AI vs AI: starts thinking, alternates agentIsBlue", () => {
    let state = gameReducer(initialState(), {
      type: "START_GAME",
      redIsHuman: false,
      blueIsHuman: false,
      redTemperature: DEFAULT_TEMPERATURE,
      blueTemperature: DEFAULT_TEMPERATURE,
    });
    expect(state.status).toBe("thinking");
    expect(state.agentIsBlue).toBe(false);

    state = gameReducer(state, { type: "AI_MOVE", cellId: 0, scores: dummyScores() });
    expect(state.cells[0]).toBe("0");
    expect(state.status).toBe("thinking");
    expect(state.agentIsBlue).toBe(true);
  });

  it("AI (red) vs human (blue): starts thinking, then idle after AI move", () => {
    let state = gameReducer(initialState(), {
      type: "START_GAME",
      redIsHuman: false,
      blueIsHuman: true,
      redTemperature: DEFAULT_TEMPERATURE,
      blueTemperature: DEFAULT_TEMPERATURE,
    });
    expect(state.status).toBe("thinking");
    expect(state.agentIsBlue).toBe(false);

    state = gameReducer(state, { type: "AI_MOVE", cellId: 0, scores: dummyScores() });
    expect(state.cells[0]).toBe("0");
    expect(state.status).toBe("idle");
  });
});

describe("human swap (pie rule)", () => {
  it("human blue can swap after AI red's first move", () => {
    let state = gameReducer(initialState(), {
      type: "START_GAME",
      redIsHuman: false,
      blueIsHuman: true,
      redTemperature: DEFAULT_TEMPERATURE,
      blueTemperature: DEFAULT_TEMPERATURE,
    });
    state = gameReducer(state, { type: "AI_MOVE", cellId: 5, scores: dummyScores() });
    expect(canSwap(state)).toBe(true);

    state = gameReducer(state, { type: "SWAP" });

    expect(state.redIsHuman).toBe(true);
    expect(state.blueIsHuman).toBe(false);
    expect(state.cells[5]).toBe("0");
    expect(state.swapUsed).toBe(true);
    expect(state.status).toBe("thinking");
    expect(state.agentIsBlue).toBe(true);
  });

  it("SWAP is a no-op when swap is no longer available", () => {
    let state = gameReducer(initialState(), {
      type: "START_GAME",
      redIsHuman: false,
      blueIsHuman: true,
      redTemperature: DEFAULT_TEMPERATURE,
      blueTemperature: DEFAULT_TEMPERATURE,
    });
    state = gameReducer(state, { type: "AI_MOVE", cellId: 5, scores: dummyScores() });
    state = gameReducer(state, { type: "SWAP" });
    const afterFirstSwap = state;

    state = gameReducer(state, { type: "SWAP" });
    expect(state).toBe(afterFirstSwap);
  });

  it("AI_MOVE double-swap is ignored", () => {
    let state = gameReducer(initialState(), {
      type: "START_GAME",
      redIsHuman: false,
      blueIsHuman: true,
      redTemperature: DEFAULT_TEMPERATURE,
      blueTemperature: DEFAULT_TEMPERATURE,
    });
    state = gameReducer(state, { type: "AI_MOVE", cellId: 5, scores: dummyScores() });
    state = gameReducer(state, { type: "SWAP" });
    const afterFirstSwap = state;

    state = gameReducer(state, { type: "AI_MOVE", cellId: 5, scores: dummyScores() });
    expect(state).toBe(afterFirstSwap);
  });

  it("canSwap becomes false once the second player plays a normal move", () => {
    let state = gameReducer(initialState(), {
      type: "START_GAME",
      redIsHuman: true,
      blueIsHuman: true,
      redTemperature: DEFAULT_TEMPERATURE,
      blueTemperature: DEFAULT_TEMPERATURE,
    });
    state = gameReducer(state, { type: "PLAYER_MOVE", cellId: 0 });
    expect(canSwap(state)).toBe(true);
    state = gameReducer(state, { type: "PLAYER_MOVE", cellId: 1 });
    expect(canSwap(state)).toBe(false);
    expect(state.swapUsed).toBe(true);
  });
});

describe("temperature", () => {
  it("defaults to DEFAULT_TEMPERATURE for both colors", () => {
    const state = initialState();
    expect(state.redTemperature).toBe(DEFAULT_TEMPERATURE);
    expect(state.blueTemperature).toBe(DEFAULT_TEMPERATURE);
  });

  it("SET_TEMPERATURE updates the named player and clamps to [0, 2]", () => {
    let state = initialState();
    state = gameReducer(state, { type: "SET_TEMPERATURE", player: "0", value: -1 });
    expect(state.redTemperature).toBe(0);
    expect(state.blueTemperature).toBe(DEFAULT_TEMPERATURE);

    state = gameReducer(state, { type: "SET_TEMPERATURE", player: "1", value: 5 });
    expect(state.blueTemperature).toBe(2);

    state = gameReducer(state, { type: "SET_TEMPERATURE", player: "0", value: 0.7 });
    expect(state.redTemperature).toBeCloseTo(0.7);
  });

  it("RESET preserves both temperatures", () => {
    let state = initialState();
    state = gameReducer(state, { type: "SET_TEMPERATURE", player: "0", value: 1.2 });
    state = gameReducer(state, { type: "SET_TEMPERATURE", player: "1", value: 0.3 });
    state = gameReducer(state, {
      type: "START_GAME",
      redIsHuman: true,
      blueIsHuman: false,
      redTemperature: 1.2,
      blueTemperature: 0.3,
    });
    state = gameReducer(state, { type: "RESET" });
    expect(state.redTemperature).toBeCloseTo(1.2);
    expect(state.blueTemperature).toBeCloseTo(0.3);
  });
});

describe("RESTART", () => {
  it("keeps player config and temperatures, skips setup", () => {
    let state = gameReducer(initialState(), {
      type: "START_GAME",
      redIsHuman: true,
      blueIsHuman: false,
      redTemperature: 0.9,
      blueTemperature: 0.1,
    });
    state = gameReducer(state, { type: "PLAYER_MOVE", cellId: 0 });
    state = gameReducer(state, { type: "AI_MOVE", cellId: 1, scores: dummyScores() });

    const restarted = gameReducer(state, { type: "RESTART" });
    expect(restarted.status).toBe("idle");
    expect(restarted.redIsHuman).toBe(true);
    expect(restarted.blueIsHuman).toBe(false);
    expect(restarted.redTemperature).toBeCloseTo(0.9);
    expect(restarted.blueTemperature).toBeCloseTo(0.1);
    expect(restarted.cells.every((c) => c === null)).toBe(true);
  });

  it("after AI swap, RESTART restores the original player setup", () => {
    // Human red vs AI blue. Human plays, AI swaps → redIsHuman/blueIsHuman flip.
    // RESTART must go back to the original setup (human red, AI blue),
    // not keep the flipped assignments.
    let state = gameReducer(initialState(), {
      type: "START_GAME",
      redIsHuman: true,
      blueIsHuman: false,
      redTemperature: DEFAULT_TEMPERATURE,
      blueTemperature: DEFAULT_TEMPERATURE,
    });
    state = gameReducer(state, { type: "PLAYER_MOVE", cellId: 0 });
    state = gameReducer(state, { type: "AI_MOVE", cellId: 0, scores: dummyScores() });
    expect(state.aiSwapped).toBe(true);
    expect(state.redIsHuman).toBe(false);
    expect(state.blueIsHuman).toBe(true);
    const aiTurnAfterSwap = state.aiTurn;

    const restarted = gameReducer(state, { type: "RESTART" });
    expect(restarted.redIsHuman).toBe(true);
    expect(restarted.blueIsHuman).toBe(false);
    expect(restarted.agentIsBlue).toBe(true);
    expect(restarted.status).toBe("idle");
    expect(restarted.swapUsed).toBe(false);
    expect(restarted.aiSwapped).toBe(false);
    expect(restarted.aiTurn).toBeGreaterThanOrEqual(aiTurnAfterSwap);
  });
});

describe("RESTORE", () => {
  it("replaces state with the snapshot", () => {
    const a = startedState();
    let b = gameReducer(a, { type: "PLAYER_MOVE", cellId: 0 });
    b = gameReducer(b, { type: "AI_MOVE", cellId: 1, scores: dummyScores() });
    const restored = gameReducer(b, { type: "RESTORE", state: a });
    expect(restored).toBe(a);
  });
});

describe("RESET", () => {
  it("returns to setup status", () => {
    let state = startedState();
    state = gameReducer(state, { type: "PLAYER_MOVE", cellId: 0 });
    state = gameReducer(state, { type: "RESET" });
    expect(state.status).toBe("setup");
    expect(state.cells.every((c) => c === null)).toBe(true);
  });
});
