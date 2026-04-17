import { useReducer, useMemo } from "react";
import { gameReducer, initialState, canSwap as canSwapSelector } from "./game/state";
import { useAI } from "./hooks/useAI";
import { findWinningPath, Player } from "./game/rules";
import HexBoard from "./components/HexBoard";
import Controls from "./components/Controls";
import PlayerSetup from "./components/PlayerSetup";
import PlayerBar from "./components/PlayerBar";

export default function App() {
  const [state, dispatch] = useReducer(gameReducer, undefined, initialState);

  useAI(state, dispatch);

  const canSwap = canSwapSelector(state);

  const winningPath = useMemo(
    () => (state.winner !== null ? findWinningPath(state.cells, state.winner) : null),
    [state.winner, state.cells]
  );

  const currentTurn: Player = useMemo(() => {
    const stones = state.cells.filter((c) => c !== null).length;
    return stones % 2 === 0 ? "0" : "1";
  }, [state.cells]);

  function handleCellClick(id: number) {
    if (canSwap && state.cells[id] !== null) {
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

      {state.status === "setup" ? (
        <PlayerSetup
          defaultRedIsHuman={state.redIsHuman}
          defaultBlueIsHuman={state.blueIsHuman}
          onStart={(redIsHuman, blueIsHuman) =>
            dispatch({ type: "START_GAME", redIsHuman, blueIsHuman })
          }
        />
      ) : (
        <>
          <PlayerBar
            redIsHuman={state.redIsHuman}
            blueIsHuman={state.blueIsHuman}
            status={state.status}
            winner={state.winner}
            currentTurn={currentTurn}
          />

          <Controls
            showRatings={state.showRatings}
            temperature={state.temperature}
            canSwap={canSwap}
            status={state.status}
            winner={state.winner}
            redIsHuman={state.redIsHuman}
            blueIsHuman={state.blueIsHuman}
            aiSwapped={state.aiSwapped}
            onToggleRatings={() => dispatch({ type: "TOGGLE_RATINGS" })}
            onSetTemperature={(value) => dispatch({ type: "SET_TEMPERATURE", value })}
            onReset={() => dispatch({ type: "RESET" })}
          />

          <HexBoard
            cells={state.cells}
            modelScores={state.modelScores}
            showRatings={state.showRatings}
            canSwap={canSwap}
            lastMove={state.lastMove}
            status={state.status}
            winningPath={winningPath}
            onCellClick={handleCellClick}
          />
        </>
      )}
    </div>
  );
}
