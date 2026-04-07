import { Cell } from "../game/rules";

// Pointy-top hexagon with unit circumradius.
// Rotated by 30° so center spacing in HexBoard tiles edge-to-edge.
const ANGLES = [30, 90, 150, 210, 270, 330].map((d) => (d * Math.PI) / 180);
const R = 1; // circumradius
const POINTS = ANGLES.map((a) => `${(R * Math.cos(a)).toFixed(4)},${(R * Math.sin(a)).toFixed(4)}`).join(" ");

const FILL: Record<string, string> = {
  "0": "rgb(251, 41, 67)",  // red player
  "1": "rgb(6, 154, 243)",  // blue player
  empty: "white",
};

interface HexCellProps {
  cx: number;
  cy: number;
  cell: Cell;
  cellId: number;
  score: number | null;
  showScore: boolean;
  onClick?: (id: number) => void;
  "data-cell-id"?: number;
}

export default function HexCell({ cx, cy, cell, cellId, score, showScore, onClick }: HexCellProps) {
  const fill = cell ? FILL[cell] : FILL.empty;
  const label = showScore && score !== null && cell === null ? score.toFixed(1) : "";

  return (
    <g
      transform={`translate(${cx.toFixed(3)},${cy.toFixed(3)})`}
      onClick={onClick ? () => onClick(cellId) : undefined}
      style={{ cursor: onClick && cell === null ? "pointer" : "default" }}
      data-cell-id={cellId}
      data-testid="hex-cell"
    >
      <polygon
        points={POINTS}
        fill={fill}
        stroke="black"
        strokeWidth={0.05}
      />
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
