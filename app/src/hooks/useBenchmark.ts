import { useRef, useState, useCallback } from "react";
import { Cell, Player, hasWinner, findSureWinMove } from "../game/rules";
import { encodeBoard, averageOutputs } from "../game/encoding";
import { selectMove } from "../game/selectMove";
import { MODEL_FILENAME, NUM_CELLS } from "../game/constants";

export interface BenchmarkStats {
  status: "idle" | "running" | "done";
  total: number;
  completed: number;
  redWins: number;
  blueWins: number;
  totalMoves: number;
}

export interface CurrentGame {
  cells: Cell[];
  lastMove: number | null;
}

let benchmarkWorker: Worker | null = null;

function getBenchmarkWorker(): Worker {
  if (!benchmarkWorker) {
    benchmarkWorker = new Worker(new URL("../ai/modelWorker.ts", import.meta.url), {
      type: "module",
    });
  }
  return benchmarkWorker;
}

function inferMove(cells: Cell[], player: Player, temperature: number): Promise<number> {
  return new Promise((resolve, reject) => {
    const agentIsBlue = player === "1";
    const { input1, input2 } = encodeBoard(cells, agentIsBlue);
    const worker = getBenchmarkWorker();
    const modelUrl = new URL(MODEL_FILENAME, document.baseURI).href;

    function onMessage(e: MessageEvent) {
      const msg = e.data;
      if (msg.type === "RESULT_PAIR") {
        worker.removeEventListener("message", onMessage);
        const scores = averageOutputs(msg.out1, msg.out2, agentIsBlue);
        resolve(selectMove(cells, scores, player, { temperature, canSwap: false }));
      } else if (msg.type === "ERROR") {
        worker.removeEventListener("message", onMessage);
        reject(new Error(msg.message));
      }
    }

    worker.addEventListener("message", onMessage);
    worker.postMessage(
      { type: "INFER_PAIR", input1, input2, boardSize: 11, modelUrl },
      [input1.buffer, input2.buffer]
    );
  });
}

async function runSingleGame(
  redTemp: number,
  blueTemp: number,
  abortRef: { current: boolean },
  onMove: (cells: Cell[], lastMove: number) => void
): Promise<{ winner: Player; numMoves: number } | null> {
  const cells: Cell[] = new Array(NUM_CELLS).fill(null);
  let numMoves = 0;

  while (true) {
    if (abortRef.current) return null;

    const player: Player = numMoves % 2 === 0 ? "0" : "1";
    const temperature = player === "0" ? redTemp : blueTemp;

    const sureWin = findSureWinMove(cells, player);
    let move: number;
    if (sureWin !== null) {
      move = sureWin;
    } else {
      move = await inferMove(cells, player, temperature);
      if (abortRef.current) return null;
    }

    cells[move] = player;
    numMoves++;
    onMove(cells.slice(), move);

    if (hasWinner(cells)) {
      return { winner: player, numMoves };
    }
  }
}

export function useBenchmark() {
  const [stats, setStats] = useState<BenchmarkStats>({
    status: "idle",
    total: 0,
    completed: 0,
    redWins: 0,
    blueWins: 0,
    totalMoves: 0,
  });
  const [currentGame, setCurrentGame] = useState<CurrentGame | null>(null);

  const abortRef = useRef(false);

  const start = useCallback((total: number, redTemp: number, blueTemp: number) => {
    abortRef.current = false;
    setStats({ status: "running", total, completed: 0, redWins: 0, blueWins: 0, totalMoves: 0 });
    setCurrentGame(null);

    (async () => {
      let redWins = 0;
      let blueWins = 0;
      let totalMoves = 0;

      const onMove = (cells: Cell[], lastMove: number) => setCurrentGame({ cells, lastMove });

      for (let i = 0; i < total; i++) {
        if (abortRef.current) break;

        let result: { winner: Player; numMoves: number } | null;
        try {
          result = await runSingleGame(redTemp, blueTemp, abortRef, onMove);
        } catch {
          break;
        }

        if (result === null || abortRef.current) break;

        redWins += result.winner === "0" ? 1 : 0;
        blueWins += result.winner === "1" ? 1 : 0;
        totalMoves += result.numMoves;

        const completed = i + 1;
        setStats({
          status: completed === total ? "done" : "running",
          total,
          completed,
          redWins,
          blueWins,
          totalMoves,
        });
      }

      setStats((s) => (s.status === "running" ? { ...s, status: "done" } : s));
    })();
  }, []);

  const stop = useCallback(() => {
    abortRef.current = true;
    setStats((s) => ({ ...s, status: "done" }));
  }, []);

  return { stats, currentGame, start, stop };
}
