import { NUM_CELLS } from "./constants";
import { Cell, Player, hasWinner } from "./rules";

export interface GameState {
  cells: Cell[];
  redIsHuman: boolean;
  blueIsHuman: boolean;
  agentIsBlue: boolean; // which player the AI acts as on the current "thinking" turn
  aiSwapped: boolean;
  winner: Player | null;
  lastMove: number | null;
  modelScores: (number | null)[];
  showRatings: boolean;
  status: "setup" | "idle" | "thinking" | "gameover";
}

export type GameAction =
  | { type: "START_GAME"; redIsHuman: boolean; blueIsHuman: boolean }
  | { type: "PLAYER_MOVE"; cellId: number }
  | { type: "SWAP" }
  | { type: "AI_MOVE"; cellId: number; scores: Float32Array }
  | { type: "AI_SURE_WIN"; cellId: number }
  | { type: "TOGGLE_RATINGS" }
  | { type: "RESET" };

export function initialState(): GameState {
  return {
    cells: Array<Cell>(NUM_CELLS).fill(null),
    redIsHuman: true,
    blueIsHuman: false,
    agentIsBlue: true,
    aiSwapped: false,
    winner: null,
    lastMove: null,
    modelScores: Array<null>(NUM_CELLS).fill(null),
    showRatings: false,
    status: "setup",
  };
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

/**
 * After placing a stone, determine the next status and agentIsBlue.
 * nextPlayer is derived from the stone count: even → red ("0"), odd → blue ("1").
 */
function afterMoveTransition(
  cells: Cell[],
  winner: Player | null,
  redIsHuman: boolean,
  blueIsHuman: boolean,
  currentAgentIsBlue: boolean
): Pick<GameState, "status" | "agentIsBlue"> {
  if (winner) return { status: "gameover", agentIsBlue: currentAgentIsBlue };
  const numMoves = cells.filter((c) => c !== null).length;
  const nextPlayer: Player = numMoves % 2 === 0 ? "0" : "1";
  const nextIsHuman = nextPlayer === "0" ? redIsHuman : blueIsHuman;
  if (nextIsHuman) {
    return { status: "idle", agentIsBlue: currentAgentIsBlue };
  }
  return { status: "thinking", agentIsBlue: nextPlayer === "1" };
}

export function gameReducer(state: GameState, action: GameAction): GameState {
  switch (action.type) {
    case "START_GAME": {
      const { redIsHuman, blueIsHuman } = action;
      // Red always moves first; set agentIsBlue for the first thinking turn if red is AI
      const agentIsBlue = redIsHuman; // AI is blue if red is human, AI is red (false) if red is AI
      const status = redIsHuman ? "idle" : "thinking";
      return {
        ...initialState(),
        redIsHuman,
        blueIsHuman,
        agentIsBlue,
        showRatings: state.showRatings,
        status,
      };
    }

    case "PLAYER_MOVE": {
      if (state.status !== "idle") return state;
      if (state.cells[action.cellId] !== null) return state;
      const numMoves = state.cells.filter((c) => c !== null).length;
      const currentPlayer: Player = numMoves % 2 === 0 ? "0" : "1";
      const next = placeStone({ ...state, aiSwapped: false }, action.cellId, currentPlayer);
      const transition = afterMoveTransition(
        next.cells, next.winner, state.redIsHuman, state.blueIsHuman, state.agentIsBlue
      );
      return { ...next, ...transition };
    }

    case "SWAP": {
      // Pie rule invoked by the human second player: roles flip.
      const newRedIsHuman = state.blueIsHuman;
      const newBlueIsHuman = state.redIsHuman;
      const newAgentIsBlue = !state.agentIsBlue;
      // After swap, blue ("1") moves next; check if that player is now AI
      const status = newBlueIsHuman ? "idle" : "thinking";
      return {
        ...state,
        redIsHuman: newRedIsHuman,
        blueIsHuman: newBlueIsHuman,
        agentIsBlue: newAgentIsBlue,
        aiSwapped: false,
        status,
      };
    }

    case "AI_MOVE": {
      if (state.cells[action.cellId] !== null) {
        // Pie rule: AI (second player) swaps by returning the occupied cell.
        const numMoves = state.cells.filter((c) => c !== null).length;
        if (numMoves === 1) {
          const scores = Array.from({ length: NUM_CELLS }, (_, i) => action.scores[i] ?? null);
          const newRedIsHuman = state.blueIsHuman;
          const newBlueIsHuman = state.redIsHuman;
          const newAgentIsBlue = !state.agentIsBlue;
          // After swap, blue ("1") moves next; check if that player is now AI
          const status = newBlueIsHuman ? "idle" : "thinking";
          return {
            ...state,
            redIsHuman: newRedIsHuman,
            blueIsHuman: newBlueIsHuman,
            agentIsBlue: newAgentIsBlue,
            aiSwapped: true,
            modelScores: scores,
            status,
          };
        }
        return state;
      }
      const agentPlayer: Player = state.agentIsBlue ? "1" : "0";
      const scores = Array.from({ length: NUM_CELLS }, (_, i) => action.scores[i] ?? null);
      const next = placeStone({ ...state, modelScores: scores, aiSwapped: false }, action.cellId, agentPlayer);
      const transition = afterMoveTransition(
        next.cells, next.winner, state.redIsHuman, state.blueIsHuman, state.agentIsBlue
      );
      return { ...next, ...transition };
    }

    case "AI_SURE_WIN": {
      const agentPlayer: Player = state.agentIsBlue ? "1" : "0";
      const next = placeStone(
        { ...state, modelScores: Array<null>(NUM_CELLS).fill(null), aiSwapped: false },
        action.cellId,
        agentPlayer
      );
      const transition = afterMoveTransition(
        next.cells, next.winner, state.redIsHuman, state.blueIsHuman, state.agentIsBlue
      );
      return { ...next, ...transition };
    }

    case "TOGGLE_RATINGS":
      return { ...state, showRatings: !state.showRatings };

    case "RESET":
      return { ...initialState(), showRatings: state.showRatings };

    default:
      return state;
  }
}
