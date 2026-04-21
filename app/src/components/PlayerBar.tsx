import { Player } from "../game/rules";

const RED = "oklch(0.62 0.22 25)";
const BLUE = "oklch(0.62 0.22 240)";
const RED_DIM = "oklch(0.62 0.22 25 / 0.1)";
const BLUE_DIM = "oklch(0.62 0.22 240 / 0.1)";

interface PlayerBarProps {
  redIsHuman: boolean;
  blueIsHuman: boolean;
  status: "idle" | "thinking" | "gameover";
  agentIsBlue: boolean;
  winner: Player | null;
  currentTurn: Player;
}

export default function PlayerBar({
  redIsHuman,
  blueIsHuman,
  status,
  agentIsBlue,
  winner,
  currentTurn,
}: PlayerBarProps) {
  const redActive = status !== "gameover" && currentTurn === "0";
  const blueActive = status !== "gameover" && currentTurn === "1";
  const redThinking = status === "thinking" && !agentIsBlue;
  const blueThinking = status === "thinking" && agentIsBlue;

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 12,
        flexWrap: "wrap",
      }}
    >
      <PlayerBadge
        color={RED}
        colorDim={RED_DIM}
        name="Red"
        isHuman={redIsHuman}
        active={redActive}
        won={winner === "0"}
        thinking={redThinking}
      />
      <span style={{ color: "var(--muted)", fontSize: 12, fontWeight: 500 }}>vs</span>
      <PlayerBadge
        color={BLUE}
        colorDim={BLUE_DIM}
        name="Blue"
        isHuman={blueIsHuman}
        active={blueActive}
        won={winner === "1"}
        thinking={blueThinking}
      />
    </div>
  );
}

interface PlayerBadgeProps {
  color: string;
  colorDim: string;
  name: string;
  isHuman: boolean;
  active: boolean;
  won: boolean;
  thinking: boolean;
}

function PlayerBadge({ color, colorDim, name, isHuman, active, won, thinking }: PlayerBadgeProps) {
  const highlighted = active || won;
  return (
    <div
      data-testid={`player-${name.toLowerCase()}`}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "10px 16px",
        borderRadius: 10,
        border: `1px solid ${highlighted ? color + "66" : "var(--border)"}`,
        background: highlighted ? colorDim : "var(--card)",
        transition: "all 0.2s",
        minWidth: 160,
        boxShadow: active ? `0 0 16px ${color}33` : "none",
        opacity: highlighted ? 1 : 0.5,
      }}
    >
      <div
        style={{
          width: 10,
          height: 10,
          borderRadius: "50%",
          background: color,
          flexShrink: 0,
          boxShadow: active ? `0 0 8px ${color}` : "none",
          animation: thinking ? "pulse 1.2s ease infinite" : "none",
        }}
      />
      <div style={{ lineHeight: 1.3 }}>
        <div style={{ fontWeight: 600, fontSize: 14, color: highlighted ? color : "var(--muted2)" }}>
          {name}
        </div>
        <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 1 }}>
          {isHuman ? "Human" : "AI"}
        </div>
      </div>
      {won && (
        <div
          style={{
            marginLeft: "auto",
            fontSize: 11,
            fontWeight: 600,
            color,
            letterSpacing: "0.05em",
          }}
        >
          WIN
        </div>
      )}
      {thinking && (
        <div
          style={{
            marginLeft: "auto",
            fontSize: 11,
            color: "var(--muted2)",
            animation: "pulse 1.2s ease infinite",
          }}
        >
          thinking…
        </div>
      )}
    </div>
  );
}
