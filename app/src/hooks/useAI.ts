import { useEffect, useRef, useCallback } from "react";
import { GameState, GameAction } from "../game/state";
import { encodeBoard, averageOutputs } from "../game/encoding";
import { findSureWinMove } from "../game/rules";
import { selectMove } from "../game/selectMove";
import { MODEL_FILENAME } from "../game/constants";

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

function createStateKey(state: GameState): string {
  return `${state.agentIsBlue}|${state.swapUsed ? "s" : "_"}|${state.cells.map((c) => c ?? "_").join("")}`;
}

export function useAI(state: GameState, dispatch: React.Dispatch<GameAction>) {
  const stateRef = useRef(state);
  stateRef.current = state;

  const pendingRef = useRef(false);

  const handleAITurn = useCallback(() => {
    if (pendingRef.current) return;
    pendingRef.current = true;

    const snapshot = stateRef.current;
    const { cells, agentIsBlue, swapUsed, redTemperature, blueTemperature } = snapshot;
    const temperature = agentIsBlue ? blueTemperature : redTemperature;
    const requestStateKey = createStateKey(snapshot);
    const agentPlayer = agentIsBlue ? "1" : "0";

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
        if (createStateKey(stateRef.current) !== requestStateKey) return;

        const scores = averageOutputs(msg.out1, msg.out2, agentIsBlue);
        const numMoves = cells.filter((c) => c !== null).length;
        const canSwapNow = numMoves === 1 && !swapUsed;
        const finalMove = selectMove(cells, scores, agentPlayer, {
          temperature,
          canSwap: canSwapNow,
        });

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
  }, [dispatch]);

  useEffect(() => {
    if (state.status === "thinking" && !state.paused) handleAITurn();
  }, [state.aiTurn, state.paused, handleAITurn]);

  useEffect(() => {
    if (state.stepSignal === 0) return;
    if (state.status === "thinking") handleAITurn();
    // Intentionally only depend on stepSignal: stepping runs one move on demand.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.stepSignal]);
}
