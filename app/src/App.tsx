import { useReducer } from "react";
import { gameReducer, initialState } from "./game/state";
import { useAI } from "./hooks/useAI";
import HexBoard from "./components/HexBoard";
import Controls from "./components/Controls";

export default function App() {
  const [state, dispatch] = useReducer(gameReducer, undefined, initialState);

  useAI(state, dispatch);

  function handleCellClick(id: number) {
    const numMoves = state.cells.filter((c) => c !== null).length;

    // Pie rule: on the second half-move, clicking the only occupied cell = swap
    if (numMoves === 1 && state.cells[id] !== null) {
      dispatch({ type: "SWAP" });
      return;
    }

    dispatch({ type: "PLAYER_MOVE", cellId: id });
  }

  return (
    <div className="app-container">
      <header>
        <p className="intro">
          HexHex — A reinforcement learning agent by Simon Buchholz, David Harbecke, and Pascal Van
          Cleeff.{" "}
          <a href="https://github.com/harbecke/HexHex">github.com/harbecke/HexHex</a>
        </p>
      </header>

      <Controls
        showRatings={state.showRatings}
        status={state.status}
        winner={state.winner}
        agentIsBlue={state.agentIsBlue}
        onToggleRatings={() => dispatch({ type: "TOGGLE_RATINGS" })}
        onReset={() => dispatch({ type: "RESET" })}
      />

      <HexBoard
        cells={state.cells}
        modelScores={state.modelScores}
        showRatings={state.showRatings}
        status={state.status}
        onCellClick={handleCellClick}
      />
    </div>
  );
}
