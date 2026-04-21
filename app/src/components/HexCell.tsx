import { useState } from "react";
import { Cell } from "../game/rules";

const ANGLES = [30, 90, 150, 210, 270, 330].map((d) => (d * Math.PI) / 180);
const R = 1;
const POINTS = ANGLES.map((a) => `${(R * Math.cos(a)).toFixed(4)},${(R * Math.sin(a)).toFixed(4)}`).join(" ");
const R_INNER = 0.45;
const INNER_POINTS = ANGLES.map((a) => `${(R_INNER * Math.cos(a)).toFixed(4)},${(R_INNER * Math.sin(a)).toFixed(4)}`).join(" ");
const R_SWAP = 0.65;
const SWAP_POINTS = ANGLES.map((a) => `${(R_SWAP * Math.cos(a)).toFixed(4)},${(R_SWAP * Math.sin(a)).toFixed(4)}`).join(" ");
const R_LAST = 0.85;
const LAST_POINTS = ANGLES.map((a) => `${(R_LAST * Math.cos(a)).toFixed(4)},${(R_LAST * Math.sin(a)).toFixed(4)}`).join(" ");

const RED_FILL = "oklch(0.62 0.22 25)";
const BLUE_FILL = "oklch(0.62 0.22 240)";
const EMPTY_FILL = "#ffffff";
const EMPTY_HOVER = "#f0f0f6";
const STROKE = "#0d0d12";

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
  const [hovered, setHovered] = useState(false);
  const stoneFill = cell === "0" ? RED_FILL : cell === "1" ? BLUE_FILL : null;
  const fill = stoneFill ?? (hovered && onClick ? EMPTY_HOVER : EMPTY_FILL);
  const showLabel = showScore && score !== null && (cell === null || isLastMove);
  const label = showLabel ? score.toFixed(1) : "";
  const clickable = Boolean(onClick) && (cell === null || isSwapTarget);
  const labelColor = cell === null ? "#66666f" : "white";

  return (
    <g
      transform={`translate(${cx.toFixed(3)},${cy.toFixed(3)})`}
      onClick={onClick ? () => onClick(cellId) : undefined}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{ cursor: clickable ? "pointer" : "default" }}
      data-cell-id={cellId}
      data-testid="hex-cell"
    >
      <polygon points={POINTS} fill={fill} stroke={STROKE} strokeWidth={0.07} />
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
          fillOpacity={0.5}
          stroke="none"
          style={{ pointerEvents: "none" }}
        />
      )}
      {isLastMove && cell !== null && (
        <polygon
          points={LAST_POINTS}
          fill="none"
          stroke="white"
          strokeWidth={0.14}
          style={{ pointerEvents: "none" }}
        />
      )}
      {label && (
        <text
          fontSize={0.32}
          textAnchor="middle"
          dominantBaseline="central"
          fill={labelColor}
          style={{ pointerEvents: "none", userSelect: "none", fontFamily: "var(--mono)" }}
        >
          {label}
        </text>
      )}
    </g>
  );
}
