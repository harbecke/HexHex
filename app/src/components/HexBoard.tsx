import { useMemo } from "react";
import { BOARD_SIZE } from "../game/constants";
import { posToId } from "../game/coords";
import { Cell } from "../game/rules";
import HexCell from "./HexCell";

// Pointy-top hex geometry
const HEX_W = Math.sqrt(3);
const HEX_H = 1.5;
const HEX_HALF_WIDTH = Math.sqrt(3) / 2;
const HEX_RADIUS = 1;

function hexCenter(x: number, y: number): [number, number] {
  const cx = (x + 0.5) * HEX_W + y * (HEX_W / 2);
  const cy = (y + 0.5) * HEX_H;
  return [cx, cy];
}

const RED_FILL = "rgb(251, 41, 67)";
const BLUE_FILL = "rgb(6, 154, 243)";

const ANGLES = [30, 90, 150, 210, 270, 330].map((d) => (d * Math.PI) / 180);
const BORDER_POINTS = ANGLES.map((a) => `${Math.cos(a).toFixed(4)},${Math.sin(a).toFixed(4)}`).join(" ");

function BorderHex({ cx, cy, fill }: { cx: number; cy: number; fill: string }) {
  return (
    <polygon
      points={BORDER_POINTS}
      fill={fill}
      stroke="black"
      strokeWidth={0.08}
      transform={`translate(${cx.toFixed(3)},${cy.toFixed(3)})`}
    />
  );
}

interface HexBoardProps {
  cells: Cell[];
  modelScores: (number | null)[];
  showRatings: boolean;
  canSwap: boolean;
  lastMove: number | null;
  status: "idle" | "thinking" | "gameover";
  winningPath: Set<number> | null;
  onCellClick: (id: number) => void;
}

export default function HexBoard({
  cells,
  modelScores,
  showRatings,
  canSwap,
  lastMove,
  status,
  winningPath,
  onCellClick,
}: HexBoardProps) {
  const canClick = status === "idle";
  const swapTargetId = canSwap ? cells.findIndex((c) => c !== null) : -1;

  const viewBox = useMemo(() => {
    const borderCenters: [number, number][] = [];
    for (let i = 0; i < BOARD_SIZE; i++) {
      borderCenters.push(hexCenter(i, -1), hexCenter(i, BOARD_SIZE), hexCenter(-1, i), hexCenter(BOARD_SIZE, i));
    }
    const playCenters: [number, number][] = [];
    for (let y = 0; y < BOARD_SIZE; y++) {
      for (let x = 0; x < BOARD_SIZE; x++) playCenters.push(hexCenter(x, y));
    }
    const all = [...borderCenters, ...playCenters];
    const minCx = Math.min(...all.map(([cx]) => cx)) - HEX_HALF_WIDTH;
    const maxCx = Math.max(...all.map(([cx]) => cx)) + HEX_HALF_WIDTH;
    const minCy = Math.min(...all.map(([, cy]) => cy)) - HEX_RADIUS;
    const maxCy = Math.max(...all.map(([, cy]) => cy)) + HEX_RADIUS;
    const pad = 0.25;
    return {
      vx: minCx - pad,
      vy: minCy - pad,
      vw: maxCx - minCx + 2 * pad,
      vh: maxCy - minCy + 2 * pad,
    };
  }, []);

  return (
    <svg
      data-testid="hex-board"
      viewBox={`${viewBox.vx} ${viewBox.vy} ${viewBox.vw} ${viewBox.vh}`}
      style={{ width: "100%", maxWidth: 900, display: "block", margin: "0 auto" }}
    >
      {Array.from({ length: BOARD_SIZE }, (_, i) => {
        const [tx, ty] = hexCenter(i, -1);
        const [bx, by] = hexCenter(i, BOARD_SIZE);
        const [lx, ly] = hexCenter(-1, i);
        const [rx, ry] = hexCenter(BOARD_SIZE, i);
        return (
          <g key={`border-${i}`}>
            <BorderHex cx={tx} cy={ty} fill={RED_FILL} />
            <BorderHex cx={bx} cy={by} fill={RED_FILL} />
            <BorderHex cx={lx} cy={ly} fill={BLUE_FILL} />
            <BorderHex cx={rx} cy={ry} fill={BLUE_FILL} />
          </g>
        );
      })}

      {Array.from({ length: BOARD_SIZE }, (_, y) =>
        Array.from({ length: BOARD_SIZE }, (_, x) => {
          const id = posToId(x, y);
          const [cx, cy] = hexCenter(x, y);
          return (
            <HexCell
              key={id}
              cx={cx}
              cy={cy}
              cell={cells[id]}
              cellId={id}
              score={modelScores[id]}
              showScore={showRatings}
              isLastMove={id === lastMove}
              isOnWinningPath={winningPath?.has(id) ?? false}
              isSwapTarget={id === swapTargetId}
              onClick={canClick ? onCellClick : undefined}
            />
          );
        })
      )}
    </svg>
  );
}
