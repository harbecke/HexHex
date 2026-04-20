import { NUM_CELLS } from "./constants";
import { Cell, Player, hasWinner } from "./rules";

export interface GameState {
  cells: Cell[];
  redIsHuman: boolean;
  blueIsHuman: boolean;
  /** Original config at START_GAME time; preserved across swap so RESTART restores it. */
  setupRedIsHuman: boolean;
  setupBlueIsHuman: boolean;
  redTemperature: number;
  blueTemperature: number;
  agentIsBlue: boolean;
  aiSwapped: boolean;
  swapUsed: boolean;
  winner: Player | null;
  lastMove: number | null;
  modelScores: (number | null)[];
  showRatings: boolean;
  status: "setup" | "idle" | "thinking" | "gameover";
  aiTurn: number;
}

export type GameAction =
  | {
      type: "START_GAME";
      redIsHuman: boolean;
      blueIsHuman: boolean;
      redTemperature: number;
      blueTemperature: number;
    }
  | { type: "PLAYER_MOVE"; cellId: number }
  | { type: "SWAP" }
  | { type: "AI_MOVE"; cellId: number; scores: Float32Array }
  | { type: "AI_SURE_WIN"; cellId: number }
  | { type: "TOGGLE_RATINGS" }
  | { type: "SET_TEMPERATURE"; player: Player; value: number }
  | { type: "RESET" }
  | { type: "RESTART" }
  | { type: "RESTORE"; state: GameState };

export const DEFAULT_TEMPERATURE = 0.5;

export function initialState(): GameState {
  return {
    cells: Array<Cell>(NUM_CELLS).fill(null),
    redIsHuman: true,
    blueIsHuman: false,
    setupRedIsHuman: true,
    setupBlueIsHuman: false,
    redTemperature: DEFAULT_TEMPERATURE,
    blueTemperature: DEFAULT_TEMPERATURE,
    agentIsBlue: true,
    aiSwapped: false,
    swapUsed: false,
    winner: null,
    lastMove: null,
    modelScores: Array<null>(NUM_CELLS).fill(null),
    showRatings: false,
    status: "setup",
    aiTurn: 0,
  };
}

/** Swap is offered only on the second half-move, and only once per game. */
export function canSwap(state: GameState): boolean {
  const stones = state.cells.filter((c) => c !== null).length;
  return stones === 1 && !state.swapUsed && state.status === "idle";
}

function placeStone(state: GameState, cellId: number, player: Player): GameState {
  const cells = state.cells.slice() as Cell[];
  cells[cellId] = player;
  const winner = hasWinner(cells) ? player : null;
  return {
    ...state,
    cells,
    lastMove: cellId,
    winner,
    status: winner ? "gameover" : state.status,
  };
}

function afterMoveTransition(
  cells: Cell[],
  winner: Player | null,
  redIsHuman: boolean,
  blueIsHuman: boolean,
  currentAgentIsBlue: boolean,
  currentAiTurn: number
): Pick<GameState, "status" | "agentIsBlue" | "aiTurn"> {
  if (winner) return { status: "gameover", agentIsBlue: currentAgentIsBlue, aiTurn: currentAiTurn };
  const numMoves = cells.filter((c) => c !== null).length;
  const nextPlayer: Player = numMoves % 2 === 0 ? "0" : "1";
  const nextIsHuman = nextPlayer === "0" ? redIsHuman : blueIsHuman;
  if (nextIsHuman) {
    return { status: "idle", agentIsBlue: currentAgentIsBlue, aiTurn: currentAiTurn };
  }
  return { status: "thinking", agentIsBlue: nextPlayer === "1", aiTurn: currentAiTurn + 1 };
}

function applySwap(state: GameState): Pick<
  GameState,
  "redIsHuman" | "blueIsHuman" | "agentIsBlue" | "swapUsed" | "lastMove" | "status" | "aiTurn"
> {
  const newBlueIsHuman = state.redIsHuman;
  const swappedCell = state.cells.findIndex((c) => c !== null);
  const status = newBlueIsHuman ? "idle" : "thinking";
  return {
    redIsHuman: state.blueIsHuman,
    blueIsHuman: newBlueIsHuman,
    agentIsBlue: !state.agentIsBlue,
    swapUsed: true,
    lastMove: swappedCell,
    status,
    aiTurn: status === "thinking" ? state.aiTurn + 1 : state.aiTurn,
  };
}

function startGame(
  redIsHuman: boolean,
  blueIsHuman: boolean,
  redTemperature: number,
  blueTemperature: number,
  showRatings: boolean,
  prevAiTurn: number
): GameState {
  const agentIsBlue = redIsHuman;
  const status = redIsHuman ? "idle" : "thinking";
  return {
    ...initialState(),
    redIsHuman,
    blueIsHuman,
    setupRedIsHuman: redIsHuman,
    setupBlueIsHuman: blueIsHuman,
    redTemperature,
    blueTemperature,
    agentIsBlue,
    showRatings,
    status,
    aiTurn: status === "thinking" ? prevAiTurn + 1 : prevAiTurn,
  };
}

export function gameReducer(state: GameState, action: GameAction): GameState {
  switch (action.type) {
    case "START_GAME":
      return startGame(
        action.redIsHuman,
        action.blueIsHuman,
        action.redTemperature,
        action.blueTemperature,
        state.showRatings,
        state.aiTurn
      );

    case "PLAYER_MOVE": {
      if (state.status !== "idle") return state;
      if (state.cells[action.cellId] !== null) return state;
      const numMoves = state.cells.filter((c) => c !== null).length;
      const currentPlayer: Player = numMoves % 2 === 0 ? "0" : "1";
      const swapUsed = state.swapUsed || numMoves === 1;
      const next = placeStone(
        { ...state, aiSwapped: false, swapUsed },
        action.cellId,
        currentPlayer
      );
      const transition = afterMoveTransition(
        next.cells, next.winner, state.redIsHuman, state.blueIsHuman, state.agentIsBlue, state.aiTurn
      );
      return { ...next, ...transition };
    }

    case "SWAP": {
      if (!canSwap(state)) return state;
      return { ...state, ...applySwap(state), aiSwapped: false };
    }

    case "AI_MOVE": {
      const scores = Array.from({ length: NUM_CELLS }, (_, i) => action.scores[i] ?? null);

      if (state.cells[action.cellId] !== null) {
        const numMoves = state.cells.filter((c) => c !== null).length;
        if (numMoves !== 1 || state.swapUsed) return state;
        return { ...state, ...applySwap(state), aiSwapped: true, modelScores: scores };
      }

      const agentPlayer: Player = state.agentIsBlue ? "1" : "0";
      const numMoves = state.cells.filter((c) => c !== null).length;
      const swapUsed = state.swapUsed || numMoves === 1;
      const next = placeStone(
        { ...state, modelScores: scores, aiSwapped: false, swapUsed },
        action.cellId,
        agentPlayer
      );
      const transition = afterMoveTransition(
        next.cells, next.winner, state.redIsHuman, state.blueIsHuman, state.agentIsBlue, state.aiTurn
      );
      return { ...next, ...transition };
    }

    case "AI_SURE_WIN": {
      const agentPlayer: Player = state.agentIsBlue ? "1" : "0";
      const numMoves = state.cells.filter((c) => c !== null).length;
      const swapUsed = state.swapUsed || numMoves === 1;
      const next = placeStone(
        { ...state, modelScores: Array<null>(NUM_CELLS).fill(null), aiSwapped: false, swapUsed },
        action.cellId,
        agentPlayer
      );
      const transition = afterMoveTransition(
        next.cells, next.winner, state.redIsHuman, state.blueIsHuman, state.agentIsBlue, state.aiTurn
      );
      return { ...next, ...transition };
    }

    case "TOGGLE_RATINGS":
      return { ...state, showRatings: !state.showRatings };

    case "SET_TEMPERATURE": {
      const key = action.player === "0" ? "redTemperature" : "blueTemperature";
      return { ...state, [key]: Math.max(0, Math.min(2, action.value)) };
    }

    case "RESET":
      return {
        ...initialState(),
        showRatings: state.showRatings,
        redTemperature: state.redTemperature,
        blueTemperature: state.blueTemperature,
      };

    case "RESTART":
      return startGame(
        state.setupRedIsHuman,
        state.setupBlueIsHuman,
        state.redTemperature,
        state.blueTemperature,
        state.showRatings,
        state.aiTurn
      );

    case "RESTORE":
      return action.state;

    default:
      return state;
  }
}
