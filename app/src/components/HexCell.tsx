import { Cell } from "../game/rules";

// Pointy-top hexagon with unit circumradius.
const ANGLES = [30, 90, 150, 210, 270, 330].map((d) => (d * Math.PI) / 180);
const R = 1;
const POINTS = ANGLES.map((a) => `${(R * Math.cos(a)).toFixed(4)},${(R * Math.sin(a)).toFixed(4)}`).join(" ");
const R_INNER = 0.45;
const INNER_POINTS = ANGLES.map((a) => `${(R_INNER * Math.cos(a)).toFixed(4)},${(R_INNER * Math.sin(a)).toFixed(4)}`).join(" ");
const R_SWAP = 0.65;
const SWAP_POINTS = ANGLES.map((a) => `${(R_SWAP * Math.cos(a)).toFixed(4)},${(R_SWAP * Math.sin(a)).toFixed(4)}`).join(" ");

const FILL: Record<string, string> = {
  "0": "rgb(251, 41, 67)",
  "1": "rgb(6, 154, 243)",
  empty: "white",
};

interface HexCellProps {
  cx: number;
  cy: number;
  cell: Cell;
  cellId: number;
  score: number | null;
  showScore: boolean;
  isLastMove?: boolean;
  isOnWinningPath?: boolean;
  isSwapTarget?: boolean;
  onClick?: (id: number) => void;
}

export default function HexCell({
  cx,
  cy,
  cell,
  cellId,
  score,
  showScore,
  isLastMove = false,
  isOnWinningPath = false,
  isSwapTarget = false,
  onClick,
}: HexCellProps) {
  const fill = cell ? FILL[cell] : FILL.empty;
  // Show score on empty cells or the cell of the move that was just made
  // (stale scores on older occupied cells aren't meaningful).
  const showLabel = showScore && score !== null && (cell === null || isLastMove);
  const label = showLabel ? score.toFixed(1) : "";
  const clickable = Boolean(onClick) && (cell === null || isSwapTarget);

  return (
    <g
      transform={`translate(${cx.toFixed(3)},${cy.toFixed(3)})`}
      onClick={onClick ? () => onClick(cellId) : undefined}
      style={{ cursor: clickable ? "pointer" : "default" }}
      data-cell-id={cellId}
      data-testid="hex-cell"
    >
      <polygon
        points={POINTS}
        fill={fill}
        stroke="black"
        strokeWidth={0.08}
      />
      {isSwapTarget && (
        <polygon
          points={SWAP_POINTS}
          fill="none"
          stroke="white"
          strokeWidth={0.12}
          strokeDasharray="0.2,0.15"
          style={{ pointerEvents: "none" }}
        >
          <animateTransform
            attributeName="transform"
            type="rotate"
            from="0"
            to="360"
            dur="8s"
            repeatCount="indefinite"
          />
        </polygon>
      )}
      {isOnWinningPath && (
        <polygon
          points={INNER_POINTS}
          fill="white"
          fillOpacity={0.55}
          stroke="none"
          style={{ pointerEvents: "none" }}
        />
      )}
      {isLastMove && cell !== null && (
        <circle
          r={0.18}
          fill="none"
          stroke="white"
          strokeWidth={0.08}
          style={{ pointerEvents: "none" }}
        />
      )}
      {label && (
        <text
          fontSize={0.35}
          textAnchor="middle"
          dominantBaseline="central"
          fill={fill === "white" ? "#333" : "white"}
          style={{ pointerEvents: "none", userSelect: "none" }}
        >
          {label}
        </text>
      )}
    </g>
  );
}
