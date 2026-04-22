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
  teacherMode: boolean;
  /** Scores pre-computed for the current human-to-move position (null until inference lands). */
  pendingTeacherScores: (number | null)[] | null;
  /** Scores frozen at the moment of the last human move, used to annotate that move. */
  teacherScores: (number | null)[] | null;
  /** Cell id of the last human move annotated by teacher mode. */
  teacherMoveId: number | null;
  status: "setup" | "idle" | "thinking" | "gameover";
  aiTurn: number;
  /** When true, AI turns won't auto-fire. Only meaningful in AI-vs-AI games. */
  paused: boolean;
  /** Monotonic counter — incrementing it requests one AI move regardless of `paused`. */
  stepSignal: number;
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
  | { type: "TOGGLE_TEACHER" }
  | { type: "TEACHER_SCORES"; scores: Float32Array }
  | { type: "SET_TEMPERATURE"; player: Player; value: number }
  | { type: "TOGGLE_PAUSE" }
  | { type: "STEP" }
  | { type: "RESET" }
  | { type: "RESTART" }
  | { type: "RESTORE"; state: GameState };

export const DEFAULT_TEMPERATURE = 0.3;

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
    teacherMode: false,
    pendingTeacherScores: null,
    teacherScores: null,
    teacherMoveId: null,
    status: "setup",
    aiTurn: 0,
    paused: false,
    stepSignal: 0,
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
    // After a swap numMoves is 1, so the next player is always blue.
    // The agent's color is therefore blue whenever blue is AI. Flipping
    // the previous agentIsBlue works for human-vs-AI (the two sides trade
    // roles) but breaks for AI-vs-AI, where both sides are AI and the
    // side-to-move is still blue after the swap.
    agentIsBlue: !newBlueIsHuman,
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
  teacherMode: boolean,
  prevAiTurn: number
): GameState {
  const agentIsBlue = redIsHuman;
  const status = redIsHuman ? "idle" : "thinking";
  const bothAI = !redIsHuman && !blueIsHuman;
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
    teacherMode,
    status,
    aiTurn: status === "thinking" ? prevAiTurn + 1 : prevAiTurn,
    paused: bothAI,
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
        state.teacherMode,
        state.aiTurn
      );

    case "PLAYER_MOVE": {
      if (state.status !== "idle") return state;
      if (state.cells[action.cellId] !== null) return state;
      const numMoves = state.cells.filter((c) => c !== null).length;
      const currentPlayer: Player = numMoves % 2 === 0 ? "0" : "1";
      const swapUsed = state.swapUsed || numMoves === 1;
      // Freeze pre-computed teacher scores onto this move. If inference hasn't
      // finished yet, leave teacherScores null — the annotation simply won't
      // appear (and the pending inference will become stale and be discarded).
      const teacherScores = state.teacherMode ? state.pendingTeacherScores : null;
      const teacherMoveId = state.teacherMode && teacherScores !== null ? action.cellId : null;
      const next = placeStone(
        { ...state, aiSwapped: false, swapUsed, pendingTeacherScores: null, teacherScores, teacherMoveId },
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
      return { ...state, ...applySwap(state), aiSwapped: false, pendingTeacherScores: null };
    }

    case "AI_MOVE": {
      const scores = Array.from({ length: NUM_CELLS }, (_, i) => action.scores[i] ?? null);

      if (state.cells[action.cellId] !== null) {
        const numMoves = state.cells.filter((c) => c !== null).length;
        if (numMoves !== 1 || state.swapUsed) return state;
        return {
          ...state,
          ...applySwap(state),
          aiSwapped: true,
          modelScores: scores,
          pendingTeacherScores: null,
        };
      }

      const agentPlayer: Player = state.agentIsBlue ? "1" : "0";
      const numMoves = state.cells.filter((c) => c !== null).length;
      const swapUsed = state.swapUsed || numMoves === 1;
      const next = placeStone(
        { ...state, modelScores: scores, aiSwapped: false, swapUsed, pendingTeacherScores: null },
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
        {
          ...state,
          modelScores: Array<null>(NUM_CELLS).fill(null),
          aiSwapped: false,
          swapUsed,
          pendingTeacherScores: null,
        },
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

    case "TOGGLE_TEACHER":
      return {
        ...state,
        teacherMode: !state.teacherMode,
        pendingTeacherScores: null,
        teacherScores: null,
        teacherMoveId: null,
      };

    case "TEACHER_SCORES": {
      if (!state.teacherMode) return state;
      const scores = Array.from({ length: NUM_CELLS }, (_, i) => action.scores[i] ?? null);
      return { ...state, pendingTeacherScores: scores };
    }

    case "SET_TEMPERATURE": {
      const key = action.player === "0" ? "redTemperature" : "blueTemperature";
      return { ...state, [key]: Math.max(0, Math.min(2, action.value)) };
    }

    case "TOGGLE_PAUSE":
      return { ...state, paused: !state.paused };

    case "STEP":
      return { ...state, stepSignal: state.stepSignal + 1 };

    case "RESET":
      return {
        ...initialState(),
        showRatings: state.showRatings,
        teacherMode: state.teacherMode,
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
        state.teacherMode,
        state.aiTurn
      );

    case "RESTORE":
      return action.state;

    default:
      return state;
  }
}
