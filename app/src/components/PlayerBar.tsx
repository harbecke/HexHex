import { Player } from "../game/rules";

const RED = "rgb(251, 41, 67)";
const BLUE = "rgb(6, 154, 243)";

interface PlayerBarProps {
  redIsHuman: boolean;
  blueIsHuman: boolean;
  status: "idle" | "thinking" | "gameover";
  winner: Player | null;
  currentTurn: Player; // whose move is expected next
}

export default function PlayerBar({
  redIsHuman,
  blueIsHuman,
  status,
  winner,
  currentTurn,
}: PlayerBarProps) {
  const redActive = status !== "gameover" && currentTurn === "0";
  const blueActive = status !== "gameover" && currentTurn === "1";

  return (
    <div className="player-bar">
      <PlayerBadge
        color={RED}
        name="Red"
        isHuman={redIsHuman}
        active={redActive}
        won={winner === "0"}
      />
      <span className="player-bar-vs">vs</span>
      <PlayerBadge
        color={BLUE}
        name="Blue"
        isHuman={blueIsHuman}
        active={blueActive}
        won={winner === "1"}
      />
    </div>
  );
}

interface PlayerBadgeProps {
  color: string;
  name: string;
  isHuman: boolean;
  active: boolean;
  won: boolean;
}

function PlayerBadge({ color, name, isHuman, active, won }: PlayerBadgeProps) {
  const label = isHuman ? "You" : "AI";
  return (
    <div
      className={`player-badge${active ? " active" : ""}${won ? " won" : ""}`}
      style={{ borderColor: color }}
      data-testid={`player-${name.toLowerCase()}`}
    >
      <div className="player-swatch" style={{ background: color }} />
      <div className="player-text">
        <span style={{ color }}>{name}</span>
        <span className="player-who"> · {label}</span>
      </div>
    </div>
  );
}
