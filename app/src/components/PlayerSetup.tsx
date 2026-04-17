import { useState } from "react";

interface PlayerSetupProps {
  defaultRedIsHuman: boolean;
  defaultBlueIsHuman: boolean;
  onStart: (redIsHuman: boolean, blueIsHuman: boolean) => void;
}

const RED = "rgb(251, 41, 67)";
const BLUE = "rgb(6, 154, 243)";

export default function PlayerSetup({ defaultRedIsHuman, defaultBlueIsHuman, onStart }: PlayerSetupProps) {
  const [redIsHuman, setRedIsHuman] = useState(defaultRedIsHuman);
  const [blueIsHuman, setBlueIsHuman] = useState(defaultBlueIsHuman);

  return (
    <div style={{ maxWidth: 360, margin: "2rem auto", fontFamily: "sans-serif" }}>
      <h2 style={{ marginBottom: "1.5rem" }}>Player Setup</h2>

      <PlayerRow
        label="Red"
        subtitle="1st player"
        color={RED}
        isHuman={redIsHuman}
        onChange={setRedIsHuman}
      />
      <PlayerRow
        label="Blue"
        subtitle="2nd player"
        color={BLUE}
        isHuman={blueIsHuman}
        onChange={setBlueIsHuman}
      />

      <button
        onClick={() => onStart(redIsHuman, blueIsHuman)}
        style={{ marginTop: "1.5rem", padding: "0.5rem 1.5rem", fontSize: "1rem", cursor: "pointer" }}
      >
        Start Game
      </button>
    </div>
  );
}

interface PlayerRowProps {
  label: string;
  subtitle: string;
  color: string;
  isHuman: boolean;
  onChange: (v: boolean) => void;
}

function PlayerRow({ label, subtitle, color, isHuman, onChange }: PlayerRowProps) {
  const id = label.toLowerCase();
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "1rem" }}>
      <div style={{ width: 80 }}>
        <span style={{ fontWeight: "bold", color }}>{label}</span>
        <div style={{ fontSize: "0.75rem", color: "#666" }}>{subtitle}</div>
      </div>
      <label style={{ cursor: "pointer" }}>
        <input
          type="radio"
          name={id}
          value="human"
          checked={isHuman}
          onChange={() => onChange(true)}
          style={{ marginRight: 4 }}
        />
        Human
      </label>
      <label style={{ cursor: "pointer" }}>
        <input
          type="radio"
          name={id}
          value="ai"
          checked={!isHuman}
          onChange={() => onChange(false)}
          style={{ marginRight: 4 }}
        />
        AI
      </label>
    </div>
  );
}
