import { useState } from "react";
import { Cell } from "../game/rules";
import { Suggestion } from "../game/teacher";

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
  /** Score to render on the played stone in teacher mode (regardless of showScore). */
  teacherPlayedScore?: number | null;
  /** Ghost marker on an empty cell the AI preferred. */
  teacherSuggestion?: Suggestion | null;
  /** Color used for ghost suggestions — the player who was choosing. */
  suggestionColor?: string | null;
  onClick?: (id: number) => void;
}

const SUGGESTION_OPACITY = [0.95, 0.75, 0.55];

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
  teacherPlayedScore = null,
  teacherSuggestion = null,
  suggestionColor = null,
  onClick,
}: HexCellProps) {
  const [hovered, setHovered] = useState(false);
  const stoneFill = cell === "0" ? RED_FILL : cell === "1" ? BLUE_FILL : null;
  const fill = stoneFill ?? (hovered && onClick ? EMPTY_HOVER : EMPTY_FILL);
  // Teacher-mode score on the played stone always wins over the ratings label,
  // since it's the frozen score from the player's own perspective at decision time.
  const teacherScoreVisible = teacherPlayedScore != null && cell !== null;
  const showLabel = teacherScoreVisible || (showScore && score !== null && (cell === null || isLastMove));
  const label = teacherScoreVisible
    ? teacherPlayedScore.toFixed(1)
    : showLabel && score !== null
      ? score.toFixed(1)
      : "";
  const clickable = Boolean(onClick) && (cell === null || isSwapTarget);
  const labelColor = cell === null ? "#66666f" : "white";

  const showSuggestion = teacherSuggestion !== null && cell === null && suggestionColor !== null;
  const suggestionOpacity =
    showSuggestion && teacherSuggestion
      ? (SUGGESTION_OPACITY[teacherSuggestion.rank - 1] ?? 0.4)
      : 0;

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
      {showSuggestion && teacherSuggestion && suggestionColor && (
        <g style={{ pointerEvents: "none" }} data-testid="teacher-suggestion">
          <polygon
            points={POINTS}
            fill={suggestionColor}
            fillOpacity={suggestionOpacity * 0.18}
            stroke={suggestionColor}
            strokeWidth={0.09}
            strokeOpacity={suggestionOpacity}
            strokeDasharray="0.18,0.12"
          />
          <text
            fontSize={0.5}
            textAnchor="middle"
            dominantBaseline="central"
            fill={suggestionColor}
            fillOpacity={suggestionOpacity}
            style={{ userSelect: "none", fontFamily: "var(--mono)", fontWeight: 500 }}
          >
            {teacherSuggestion.score.toFixed(1)}
          </text>
        </g>
      )}
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
