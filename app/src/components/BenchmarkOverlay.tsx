import { useState } from "react";
import { createPortal } from "react-dom";
import { useBenchmark, CurrentGame } from "../hooks/useBenchmark";
import { BOARD_SIZE } from "../game/constants";
import { Cell } from "../game/rules";

const GAME_COUNTS = [10, 20, 50, 100] as const;

// ─── Mini board geometry ──────────────────────────────────────────────────────

const HEX_W = Math.sqrt(3);
const HEX_H = 1.5;
const HEX_R = 1;
const HEX_HALF_W = HEX_W / 2;

function hexCenter(x: number, y: number): [number, number] {
  return [(x + 0.5) * HEX_W + y * HEX_HALF_W, (y + 0.5) * HEX_H];
}

const HEX_PTS = [30, 90, 150, 210, 270, 330]
  .map((d) => (d * Math.PI) / 180)
  .map((a) => [Math.cos(a), Math.sin(a)]);

function polyPoints(cx: number, cy: number): string {
  return HEX_PTS.map(([dx, dy]) => `${(cx + dx).toFixed(3)},${(cy + dy).toFixed(3)}`).join(" ");
}

// Compute a stable viewBox string at module init time
const MINI_VIEWBOX = (() => {
  const centers: [number, number][] = [];
  for (let i = 0; i < BOARD_SIZE; i++) {
    centers.push(
      hexCenter(i, -1),
      hexCenter(i, BOARD_SIZE),
      hexCenter(-1, i),
      hexCenter(BOARD_SIZE, i)
    );
  }
  for (let y = 0; y < BOARD_SIZE; y++)
    for (let x = 0; x < BOARD_SIZE; x++) centers.push(hexCenter(x, y));
  const xs = centers.map(([cx]) => cx);
  const ys = centers.map(([, cy]) => cy);
  const pad = 0.3;
  const vx = Math.min(...xs) - HEX_HALF_W - pad;
  const vy = Math.min(...ys) - HEX_R - pad;
  const vw = Math.max(...xs) - Math.min(...xs) + HEX_W + 2 * pad;
  const vh = Math.max(...ys) - Math.min(...ys) + 2 * HEX_R + 2 * pad;
  return `${vx.toFixed(3)} ${vy.toFixed(3)} ${vw.toFixed(3)} ${vh.toFixed(3)}`;
})();

const RED_FILL = "oklch(0.62 0.22 25)";
const BLUE_FILL = "oklch(0.62 0.22 240)";
const EMPTY_FILL = "var(--surface)";
const STROKE = "#0d0d12";

function MiniBoardSvg({ game }: { game: CurrentGame }) {
  return (
    <svg viewBox={MINI_VIEWBOX} style={{ width: "100%", display: "block" }}>
      {Array.from({ length: BOARD_SIZE }, (_, i) => {
        const [tx, ty] = hexCenter(i, -1);
        const [bx, by] = hexCenter(i, BOARD_SIZE);
        const [lx, ly] = hexCenter(-1, i);
        const [rx, ry] = hexCenter(BOARD_SIZE, i);
        return (
          <g key={i}>
            <polygon points={polyPoints(tx, ty)} fill={RED_FILL} fillOpacity={0.85} stroke={STROKE} strokeWidth={0.06} />
            <polygon points={polyPoints(bx, by)} fill={RED_FILL} fillOpacity={0.85} stroke={STROKE} strokeWidth={0.06} />
            <polygon points={polyPoints(lx, ly)} fill={BLUE_FILL} fillOpacity={0.85} stroke={STROKE} strokeWidth={0.06} />
            <polygon points={polyPoints(rx, ry)} fill={BLUE_FILL} fillOpacity={0.85} stroke={STROKE} strokeWidth={0.06} />
          </g>
        );
      })}
      {(game.cells as Cell[]).map((cell, id) => {
        const x = id % BOARD_SIZE;
        const y = Math.floor(id / BOARD_SIZE);
        const [cx, cy] = hexCenter(x, y);
        const fill = cell === "0" ? RED_FILL : cell === "1" ? BLUE_FILL : EMPTY_FILL;
        const isLast = id === game.lastMove;
        return (
          <g key={id}>
            <polygon points={polyPoints(cx, cy)} fill={fill} stroke={STROKE} strokeWidth={0.06} />
            {isLast && <circle cx={cx} cy={cy} r={0.32} fill="white" fillOpacity={0.65} />}
          </g>
        );
      })}
    </svg>
  );
}

// ─── Overlay ──────────────────────────────────────────────────────────────────

interface Props {
  onClose: () => void;
}

export default function BenchmarkOverlay({ onClose }: Props) {
  const [gameCount, setGameCount] = useState<number>(20);
  const [temperature, setTemperature] = useState(0.3);
  const { stats, currentGame, start, stop } = useBenchmark();

  const isRunning = stats.status === "running";
  const hasSamples = stats.completed > 0;

  function handleRun() {
    if (isRunning) {
      stop();
    } else {
      start(gameCount, temperature, temperature);
    }
  }

  const progress = stats.total > 0 ? stats.completed / stats.total : 0;
  const redPct = hasSamples ? (stats.redWins / stats.completed) * 100 : 0;
  const bluePct = hasSamples ? (stats.blueWins / stats.completed) * 100 : 0;
  const avgMoves = hasSamples ? Math.round(stats.totalMoves / stats.completed) : 0;

  return createPortal(
    <div
      onClick={(e) => e.target === e.currentTarget && onClose()}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.45)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 200,
        padding: "20px 16px",
        overflowY: "auto",
      }}
    >
      <div
        style={{
          background: "var(--card)",
          border: "1px solid var(--border)",
          borderRadius: 16,
          width: "100%",
          maxWidth: 420,
          overflow: "hidden",
          animation: "fadeUp 0.2s ease",
          margin: "auto",
        }}
      >
        {/* Header */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "20px 20px 0",
          }}
        >
          <div>
            <div style={{ fontWeight: 600, fontSize: 16, letterSpacing: "-0.01em" }}>AI Battle</div>
            <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 2 }}>
              Run AI vs AI games and collect win statistics
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: "none",
              border: "none",
              color: "var(--muted)",
              cursor: "pointer",
              fontSize: 20,
              lineHeight: 1,
              padding: "4px 6px",
              borderRadius: 6,
              fontFamily: "var(--font)",
              transition: "color 0.15s",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "var(--text)")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "var(--muted)")}
          >
            ×
          </button>
        </div>

        {/* Config */}
        <div style={{ padding: "20px 20px 0", display: "flex", flexDirection: "column", gap: 14 }}>
          <ConfigRow label="Games">
            {GAME_COUNTS.map((n) => (
              <PillButton key={n} active={gameCount === n} disabled={isRunning} onClick={() => setGameCount(n)}>
                {n}
              </PillButton>
            ))}
          </ConfigRow>

          <div>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: 6,
              }}
            >
              <span
                style={{
                  fontSize: 12,
                  color: "var(--muted)",
                  letterSpacing: "0.05em",
                  textTransform: "uppercase",
                  fontWeight: 500,
                }}
              >
                Temperature
              </span>
              <span style={{ fontSize: 13, fontFamily: "var(--mono)", color: "var(--text)", fontWeight: 500 }}>
                {temperature.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={2}
              step={0.05}
              value={temperature}
              disabled={isRunning}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              style={{ width: "100%", accentColor: "var(--text)", opacity: isRunning ? 0.5 : 1 }}
            />
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 3 }}>
              <span style={{ fontSize: 11, color: "var(--muted)" }}>Sharp</span>
              <span style={{ fontSize: 11, color: "var(--muted)" }}>Random</span>
            </div>
          </div>
        </div>

        {/* Run / Stop button */}
        <div style={{ padding: "16px 20px 0" }}>
          <button
            onClick={handleRun}
            style={{
              width: "100%",
              padding: "11px 20px",
              border: "none",
              borderRadius: 10,
              background: isRunning ? "var(--surface)" : "var(--text)",
              color: isRunning ? "var(--muted2)" : "var(--bg)",
              fontSize: 14,
              fontWeight: 600,
              fontFamily: "var(--font)",
              cursor: "pointer",
              letterSpacing: "-0.01em",
              transition: "opacity 0.15s",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.opacity = "0.85")}
            onMouseLeave={(e) => (e.currentTarget.style.opacity = "1")}
          >
            {isRunning ? "Stop" : `Run ${gameCount} games`}
          </button>
        </div>

        {/* Progress */}
        {stats.status !== "idle" && (
          <div style={{ padding: "14px 20px 0" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
              <span style={{ fontSize: 12, color: "var(--muted)" }}>
                {isRunning ? "Running…" : "Done"}
              </span>
              <span style={{ fontSize: 12, color: "var(--muted2)", fontFamily: "var(--mono)" }}>
                {stats.completed} / {stats.total}
              </span>
            </div>
            <div
              style={{ height: 4, background: "var(--surface)", borderRadius: 4, overflow: "hidden" }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${progress * 100}%`,
                  background: "var(--text)",
                  borderRadius: 4,
                  transition: "width 0.3s ease",
                }}
              />
            </div>
          </div>
        )}

        {/* Live game board */}
        {isRunning && currentGame && (
          <div style={{ padding: "14px 20px 0" }}>
            <MiniBoardSvg game={currentGame} />
          </div>
        )}

        {/* Stats */}
        {hasSamples && (
          <div style={{ padding: "14px 20px 0" }}>
            <div style={{ display: "flex", gap: 10 }}>
              <StatBlock color={RED_FILL} label="Red" wins={stats.redWins} pct={redPct} />
              <StatBlock color={BLUE_FILL} label="Blue" wins={stats.blueWins} pct={bluePct} />
            </div>
            <div
              style={{
                marginTop: 10,
                padding: "10px 14px",
                background: "var(--surface)",
                borderRadius: 8,
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <span style={{ fontSize: 12, color: "var(--muted)" }}>Avg game length</span>
              <span style={{ fontSize: 14, fontWeight: 600, fontFamily: "var(--mono)" }}>
                {avgMoves} moves
              </span>
            </div>
          </div>
        )}

        {/* Run again */}
        {stats.status === "done" && hasSamples && (
          <div style={{ padding: "12px 20px 0" }}>
            <button
              onClick={() => start(gameCount, temperature, temperature)}
              style={{
                width: "100%",
                padding: "9px 20px",
                border: "1px solid var(--border)",
                borderRadius: 10,
                background: "transparent",
                color: "var(--muted2)",
                fontSize: 13,
                fontWeight: 500,
                fontFamily: "var(--font)",
                cursor: "pointer",
                transition: "color 0.15s, border-color 0.15s",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.color = "var(--text)";
                e.currentTarget.style.borderColor = "var(--border2)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.color = "var(--muted2)";
                e.currentTarget.style.borderColor = "var(--border)";
              }}
            >
              Run again
            </button>
          </div>
        )}

        <div style={{ height: 20 }} />
      </div>
    </div>,
    document.body
  );
}

function ConfigRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <span
        style={{
          fontSize: 12,
          color: "var(--muted)",
          letterSpacing: "0.05em",
          textTransform: "uppercase",
          fontWeight: 500,
          minWidth: 90,
        }}
      >
        {label}
      </span>
      <div style={{ display: "flex", gap: 6 }}>{children}</div>
    </div>
  );
}

function PillButton({
  active,
  disabled,
  onClick,
  children,
}: {
  active: boolean;
  disabled: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: "5px 12px",
        border: active ? "1px solid var(--border2)" : "1px solid var(--border)",
        borderRadius: 7,
        background: active ? "var(--text)" : "transparent",
        color: active ? "var(--bg)" : "var(--muted2)",
        fontSize: 13,
        fontWeight: active ? 600 : 400,
        fontFamily: "var(--font)",
        cursor: disabled ? "default" : "pointer",
        opacity: disabled && !active ? 0.5 : 1,
        transition: "all 0.12s",
      }}
    >
      {children}
    </button>
  );
}

function StatBlock({
  color,
  label,
  wins,
  pct,
}: {
  color: string;
  label: string;
  wins: number;
  pct: number;
}) {
  return (
    <div
      style={{
        flex: 1,
        padding: "12px 14px",
        background: "var(--surface)",
        borderRadius: 10,
        display: "flex",
        flexDirection: "column",
        gap: 2,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
        <div style={{ width: 10, height: 10, borderRadius: "50%", background: color }} />
        <span style={{ fontSize: 12, color: "var(--muted)", fontWeight: 500 }}>{label}</span>
      </div>
      <div style={{ fontSize: 28, fontWeight: 700, color, lineHeight: 1, letterSpacing: "-0.02em" }}>
        {pct.toFixed(1)}%
      </div>
      <div style={{ fontSize: 12, color: "var(--muted2)", marginTop: 2 }}>
        {wins} win{wins !== 1 ? "s" : ""}
      </div>
    </div>
  );
}
