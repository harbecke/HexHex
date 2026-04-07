import { NUM_CELLS } from "./constants";
import { Cell, Player, hasWinner } from "./rules";

export interface GameState {
  cells: Cell[];
  agentIsBlue: boolean; // true → human is red, agent is blue
  aiSwapped: boolean;
  winner: Player | null;
  lastMove: number | null;
  modelScores: (number | null)[];
  showRatings: boolean;
  status: "idle" | "thinking" | "gameover";
}

export type GameAction =
  | { type: "PLAYER_MOVE"; cellId: number }
  | { type: "SWAP" }
  | { type: "AI_MOVE"; cellId: number; scores: Float32Array }
  | { type: "AI_SURE_WIN"; cellId: number }
  | { type: "TOGGLE_RATINGS" }
  | { type: "RESET" };

export function initialState(): GameState {
  return {
    cells: Array<Cell>(NUM_CELLS).fill(null),
    agentIsBlue: true,
    aiSwapped: false,
    winner: null,
    lastMove: null,
    modelScores: Array<null>(NUM_CELLS).fill(null),
    showRatings: false,
    status: "idle",
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

export function gameReducer(state: GameState, action: GameAction): GameState {
  switch (action.type) {
    case "PLAYER_MOVE": {
      if (state.status !== "idle") return state;
      if (state.cells[action.cellId] !== null) return state;
      const humanPlayer: Player = state.agentIsBlue ? "0" : "1";
      const next = placeStone({ ...state, aiSwapped: false }, action.cellId, humanPlayer);
      return next.winner ? next : { ...next, status: "thinking" };
    }

    case "SWAP": {
      // Pie rule: human takes the agent's first stone
      return { ...state, agentIsBlue: !state.agentIsBlue, aiSwapped: false, status: "thinking" };
    }

    case "AI_MOVE": {
      if (state.cells[action.cellId] !== null) {
        // Pie rule: AI (second player) can choose the already occupied first cell to swap.
        // The stone color stays as-is; only sides/colors assigned to human/agent flip.
        const numMoves = state.cells.filter((c) => c !== null).length;
        if (numMoves === 1) {
          const scores = Array.from({ length: NUM_CELLS }, (_, i) => action.scores[i] ?? null);
          return {
            ...state,
            agentIsBlue: !state.agentIsBlue,
            aiSwapped: true,
            modelScores: scores,
            status: "idle",
          };
        }
        return state;
      }
      const agentPlayer: Player = state.agentIsBlue ? "1" : "0";
      const scores = Array.from({ length: NUM_CELLS }, (_, i) => action.scores[i] ?? null);
      const next = placeStone({ ...state, modelScores: scores, aiSwapped: false }, action.cellId, agentPlayer);
      return next.winner ? next : { ...next, status: "idle" };
    }

    case "AI_SURE_WIN": {
      const agentPlayer: Player = state.agentIsBlue ? "1" : "0";
      const next = placeStone(
        { ...state, modelScores: Array<null>(NUM_CELLS).fill(null), aiSwapped: false },
        action.cellId,
        agentPlayer
      );
      return next.winner ? next : { ...next, status: "idle" };
    }

    case "TOGGLE_RATINGS":
      return { ...state, showRatings: !state.showRatings };

    case "RESET":
      return initialState();

    default:
      return state;
  }
}
