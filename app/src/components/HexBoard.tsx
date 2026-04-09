import { BOARD_SIZE } from "../game/constants";
import { posToId } from "../game/coords";
import { Cell } from "../game/rules";
import HexCell from "./HexCell";

// Pointy-top hex geometry
// Horizontal center spacing between neighboring columns
const HEX_W = Math.sqrt(3); // ≈ 1.732
// Vertical center spacing between neighboring rows
const HEX_H = 1.5;
const HEX_HALF_WIDTH = Math.sqrt(3) / 2;
const HEX_RADIUS = 1;

// The board is sheared: row y is offset by y * HEX_W/2 to the right.
function hexCenter(x: number, y: number): [number, number] {
  const cx = (x + 0.5) * HEX_W + y * (HEX_W / 2);
  const cy = (y + 0.5) * HEX_H;
  return [cx, cy];
}

// Border hex centers (one row above, below, left, right of the board)
function borderCenterTop(x: number): [number, number] {
  return hexCenter(x, -1);
}
function borderCenterBottom(x: number): [number, number] {
  return hexCenter(x, BOARD_SIZE);
}
function borderCenterLeft(y: number): [number, number] {
  return hexCenter(-1, y);
}
function borderCenterRight(y: number): [number, number] {
  return hexCenter(BOARD_SIZE, y);
}

const RED_FILL = "rgb(251, 41, 67)";
const BLUE_FILL = "rgb(6, 154, 243)";

// Pointy-top hex points (same as HexCell but inline for border cells)
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
  aiSwapped: boolean;
  status: "idle" | "thinking" | "gameover";
  winningPath: Set<number> | null;
  onCellClick: (id: number) => void;
}

export default function HexBoard({
  cells,
  modelScores,
  showRatings,
  aiSwapped,
  status,
  winningPath,
  onCellClick,
}: HexBoardProps) {
  const canClick = status === "idle";
  const numMoves = cells.filter((c) => c !== null).length;
  const allowScoreOnOccupied = aiSwapped && numMoves === 1;

  // Compute SVG viewBox from all rendered hex centers (board + borders).
  const borderCenters = Array.from({ length: BOARD_SIZE }, (_, i) => [
    borderCenterTop(i),
    borderCenterBottom(i),
    borderCenterLeft(i),
    borderCenterRight(i),
  ]).flat();
  const playCenters = Array.from({ length: BOARD_SIZE }, (_, y) =>
    Array.from({ length: BOARD_SIZE }, (_, x) => hexCenter(x, y))
  ).flat();
  const allCenters = [...borderCenters, ...playCenters];
  const minCx = Math.min(...allCenters.map(([cx]) => cx)) - HEX_HALF_WIDTH;
  const maxCx = Math.max(...allCenters.map(([cx]) => cx)) + HEX_HALF_WIDTH;
  const minCy = Math.min(...allCenters.map(([, cy]) => cy)) - HEX_RADIUS;
  const maxCy = Math.max(...allCenters.map(([, cy]) => cy)) + HEX_RADIUS;
  const pad = 0.25;
  const vx = minCx - pad;
  const vy = minCy - pad;
  const vw = maxCx - minCx + 2 * pad;
  const vh = maxCy - minCy + 2 * pad;

  return (
    <svg
      data-testid="hex-board"
      viewBox={`${vx} ${vy} ${vw} ${vh}`}
      style={{ width: "100%", maxWidth: 900, display: "block", margin: "0 auto" }}
    >
      {/* Border hexagons */}
      {Array.from({ length: BOARD_SIZE }, (_, i) => (
        <g key={`border-${i}`}>
          <BorderHex {...(([cx, cy]) => ({ cx, cy }))(borderCenterTop(i))} fill={RED_FILL} />
          <BorderHex {...(([cx, cy]) => ({ cx, cy }))(borderCenterBottom(i))} fill={RED_FILL} />
          <BorderHex {...(([cx, cy]) => ({ cx, cy }))(borderCenterLeft(i))} fill={BLUE_FILL} />
          <BorderHex {...(([cx, cy]) => ({ cx, cy }))(borderCenterRight(i))} fill={BLUE_FILL} />
        </g>
      ))}

      {/* Play cells */}
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
              allowScoreOnOccupied={allowScoreOnOccupied}
              isOnWinningPath={winningPath?.has(id) ?? false}
              onClick={canClick ? onCellClick : undefined}
            />
          );
        })
      )}
    </svg>
  );
}
