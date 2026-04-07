import { useEffect, useRef, useCallback } from "react";
import { GameState, GameAction } from "../game/state";
import { encodeBoard, averageOutputs } from "../game/encoding";
import { findSureWinMove } from "../game/rules";
import { NUM_CELLS, MODEL_FILENAME } from "../game/constants";

function getModelUrl(): string {
  return new URL(MODEL_FILENAME, document.baseURI).href;
}

let workerInstance: Worker | null = null;

function getWorker(): Worker {
  if (!workerInstance) {
    workerInstance = new Worker(new URL("../ai/modelWorker.ts", import.meta.url), {
      type: "module",
    });
  }
  return workerInstance;
}

export function useAI(state: GameState, dispatch: React.Dispatch<GameAction>) {
  // Keep a ref to the latest state so the callback doesn't need state as a dep
  // (prevents re-creating the callback — and re-triggering the effect — on every render)
  const stateRef = useRef(state);
  stateRef.current = state;

  const pendingRef = useRef(false);

  const handleAITurn = useCallback(() => {
    if (pendingRef.current) return;
    pendingRef.current = true;

    const { cells, agentIsBlue } = stateRef.current;
    const agentPlayer = agentIsBlue ? "1" : "0";

    // Check for a forced win first (synchronous, fast)
    const sureWin = findSureWinMove(cells, agentPlayer);
    if (sureWin !== null) {
      pendingRef.current = false;
      dispatch({ type: "AI_SURE_WIN", cellId: sureWin });
      return;
    }

    const { input1, input2 } = encodeBoard(cells, agentIsBlue);
    const worker = getWorker();

    function onMessage(e: MessageEvent) {
      const msg = e.data;
      if (msg.type === "RESULT_PAIR") {
        worker.removeEventListener("message", onMessage);
        pendingRef.current = false;

        const scores = averageOutputs(msg.out1, msg.out2, agentIsBlue);

        let best = -1;
        let bestScore = -Infinity;
        const numMoves = cells.filter((c) => c !== null).length;

        for (let i = 0; i < NUM_CELLS; i++) {
          if (numMoves > 1 && cells[i] !== null) continue;
          if (scores[i] > bestScore) {
            bestScore = scores[i];
            best = i;
          }
        }

        // Block human's sure-win if agent's best move allows it
        const humanPlayer = agentIsBlue ? "0" : "1";
        const testCells = cells.slice();
        testCells[best] = agentPlayer as "0" | "1";
        const humanWin = findSureWinMove(testCells, humanPlayer as "0" | "1");
        const finalMove = humanWin !== null ? humanWin : best;

        dispatch({ type: "AI_MOVE", cellId: finalMove, scores });
      } else if (msg.type === "ERROR") {
        worker.removeEventListener("message", onMessage);
        pendingRef.current = false;
        console.error("AI worker error:", msg.message);
      }
    }

    worker.addEventListener("message", onMessage);
    worker.postMessage(
      { type: "INFER_PAIR", input1, input2, boardSize: 11, modelUrl: getModelUrl() },
      [input1.buffer, input2.buffer]
    );
  }, [dispatch]); // dispatch is stable — callback never recreated

  useEffect(() => {
    if (state.status === "thinking") {
      handleAITurn();
    }
  }, [state.status, handleAITurn]);
}
