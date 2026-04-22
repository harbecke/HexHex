import { useReducer, useMemo, useRef, useState, useEffect, useCallback } from "react";
import { gameReducer, initialState, canSwap as canSwapSelector, GameState } from "./game/state";
import { useAI } from "./hooks/useAI";
import { findWinningPath, Player } from "./game/rules";
import { classifyMoveQuality } from "./game/teacher";
import HexBoard from "./components/HexBoard";
import Controls from "./components/Controls";
import PlayerSetup from "./components/PlayerSetup";
import PlayerBar from "./components/PlayerBar";
import StatusBanner from "./components/StatusBanner";

const BOARD_MAX_WIDTH = 740;

export default function App() {
  const [state, dispatch] = useReducer(gameReducer, undefined, initialState);

  useAI(state, dispatch);

  const historyRef = useRef<GameState[]>([]);
  const [historyLen, setHistoryLen] = useState(0);

  const clearHistory = useCallback(() => {
    historyRef.current = [];
    setHistoryLen(0);
  }, []);

  const pushHistory = useCallback(() => {
    historyRef.current = [...historyRef.current, state];
    setHistoryLen((l) => l + 1);
  }, [state]);

  const handleUndo = useCallback(() => {
    const hist = historyRef.current;
    if (hist.length === 0) return;
    const prev = hist[hist.length - 1];
    historyRef.current = hist.slice(0, -1);
    setHistoryLen((l) => l - 1);
    dispatch({ type: "RESTORE", state: prev });
  }, []);

  const handleReset = useCallback(() => {
    clearHistory();
    dispatch({ type: "RESET" });
  }, [clearHistory]);

  const handleRestart = useCallback(() => {
    clearHistory();
    dispatch({ type: "RESTART" });
  }, [clearHistory]);

  const canSwap = canSwapSelector(state);
  const canUndo = historyLen > 0 && state.status !== "setup" && state.status !== "thinking";
  const bothAI = !state.redIsHuman && !state.blueIsHuman;
  const canStep = bothAI && state.paused && state.status === "thinking";

  const winningPath = useMemo(
    () => (state.winner !== null ? findWinningPath(state.cells, state.winner) : null),
    [state.winner, state.cells]
  );

  const currentTurn: Player = useMemo(() => {
    const stones = state.cells.filter((c) => c !== null).length;
    return stones % 2 === 0 ? "0" : "1";
  }, [state.cells]);

  const teacherQuality = useMemo(
    () =>
      state.teacherMode && state.teacherMoveId !== null
        ? classifyMoveQuality(state.teacherScores, state.cells, state.teacherMoveId)
        : null,
    [state.teacherMode, state.teacherScores, state.teacherMoveId, state.cells]
  );

  // Show a "analyzing…" hint when teacher mode is on but we have nothing to
  // render yet — typically the first cold-start inference, or when the user
  // moves faster than the worker can respond.
  const teacherLoading =
    state.teacherMode &&
    teacherQuality === null &&
    state.pendingTeacherScores === null &&
    state.status === "idle";

  const showTeacher = state.redIsHuman || state.blueIsHuman;

  // Global hotkeys — skip when an input/textarea has focus.
  useEffect(() => {
    if (state.status === "setup") return;
    function onKey(e: KeyboardEvent) {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      const k = e.key.toLowerCase();
      if (k === "u" && canUndo) {
        e.preventDefault();
        handleUndo();
      } else if (k === "n") {
        e.preventDefault();
        handleReset();
      } else if (k === "r") {
        e.preventDefault();
        handleRestart();
      } else if (k === "s") {
        e.preventDefault();
        dispatch({ type: "TOGGLE_RATINGS" });
      } else if (k === "t" && showTeacher) {
        e.preventDefault();
        dispatch({ type: "TOGGLE_TEACHER" });
      } else if (k === "p" && bothAI) {
        e.preventDefault();
        dispatch({ type: "TOGGLE_PAUSE" });
      } else if (k === "." && canStep) {
        e.preventDefault();
        dispatch({ type: "STEP" });
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [state.status, canUndo, bothAI, canStep, showTeacher, handleUndo, handleReset, handleRestart]);

  function handleCellClick(id: number) {
    if (canSwap && state.cells[id] !== null) {
      pushHistory();
      dispatch({ type: "SWAP" });
      return;
    }
    pushHistory();
    dispatch({ type: "PLAYER_MOVE", cellId: id });
  }

  if (state.status === "setup") {
    return (
      <PlayerSetup
        defaultRedIsHuman={state.redIsHuman}
        defaultBlueIsHuman={state.blueIsHuman}
        defaultRedTemperature={state.redTemperature}
        defaultBlueTemperature={state.blueTemperature}
        onStart={(redIsHuman, blueIsHuman, redTemperature, blueTemperature) =>
          dispatch({
            type: "START_GAME",
            redIsHuman,
            blueIsHuman,
            redTemperature,
            blueTemperature,
          })
        }
      />
    );
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        minHeight: "100vh",
        alignItems: "center",
        padding: "20px 16px 32px",
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: BOARD_MAX_WIDTH,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 20,
          animation: "fadeUp 0.35s ease",
        }}
      >
        <button
          data-testid="home"
          onClick={handleReset}
          title="Back to setup (N)"
          style={{
            background: "none",
            border: "none",
            color: "var(--muted)",
            fontSize: 22,
            fontWeight: 300,
            letterSpacing: "-0.04em",
            cursor: "pointer",
            fontFamily: "var(--font)",
            padding: 0,
            transition: "color 0.15s",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.color = "var(--text)")}
          onMouseLeave={(e) => (e.currentTarget.style.color = "var(--muted)")}
        >
          HexHex
        </button>

        <PlayerBar
          redIsHuman={state.redIsHuman}
          blueIsHuman={state.blueIsHuman}
          status={state.status}
          agentIsBlue={state.agentIsBlue}
          winner={state.winner}
          currentTurn={currentTurn}
        />

        <div style={{ width: 48 }} />
      </div>

      <div
        style={{
          width: "100%",
          maxWidth: BOARD_MAX_WIDTH,
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <HexBoard
          cells={state.cells}
          modelScores={state.modelScores}
          showRatings={state.showRatings}
          canSwap={canSwap}
          lastMove={state.lastMove}
          status={state.status}
          winningPath={winningPath}
          teacherMode={state.teacherMode}
          teacherScores={state.teacherScores}
          teacherMoveId={state.teacherMoveId}
          onCellClick={handleCellClick}
        />
      </div>

      <div
        style={{
          marginTop: 20,
          width: "100%",
          maxWidth: BOARD_MAX_WIDTH,
          animation: "fadeUp 0.35s ease 0.1s both",
          display: "flex",
          flexDirection: "column",
          gap: 12,
          alignItems: "center",
        }}
      >
        <StatusBanner
          status={state.status}
          winner={state.winner}
          redIsHuman={state.redIsHuman}
          blueIsHuman={state.blueIsHuman}
          canSwap={canSwap}
          aiSwapped={state.aiSwapped}
          teacherQuality={teacherQuality}
          teacherLoading={teacherLoading}
        />
        <Controls
          showRatings={state.showRatings}
          teacherMode={state.teacherMode}
          showTeacher={showTeacher}
          canUndo={canUndo}
          bothAI={bothAI}
          paused={state.paused}
          canStep={canStep}
          onUndo={handleUndo}
          onReset={handleReset}
          onRestart={handleRestart}
          onToggleRatings={() => dispatch({ type: "TOGGLE_RATINGS" })}
          onToggleTeacher={() => dispatch({ type: "TOGGLE_TEACHER" })}
          onTogglePause={() => dispatch({ type: "TOGGLE_PAUSE" })}
          onStep={() => dispatch({ type: "STEP" })}
        />
        {state.showRatings && (
          <div
            data-testid="ratings-info"
            style={{
              fontSize: 12,
              color: "var(--muted2)",
              textAlign: "center",
              maxWidth: 560,
              lineHeight: 1.6,
              padding: "10px 16px",
              borderRadius: 10,
              border: "1px solid var(--border)",
              background: "var(--card)",
              animation: "fadeUp 0.25s ease",
            }}
          >
            <strong style={{ color: "var(--text)", fontWeight: 600 }}>Move values:</strong>{" "}
            logits of the AI's estimated win probability after each move, from the mover's
            perspective. Higher is better; sigmoid(score) ≈ P(win) (0 ≈ 50%, +2 ≈ ~88%).
          </div>
        )}
      </div>
    </div>
  );
}
