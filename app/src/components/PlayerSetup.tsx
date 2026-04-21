import { useState, useRef, CSSProperties } from "react";
import { createPortal } from "react-dom";

interface PlayerSetupProps {
  defaultRedIsHuman: boolean;
  defaultBlueIsHuman: boolean;
  defaultRedTemperature: number;
  defaultBlueTemperature: number;
  onStart: (
    redIsHuman: boolean,
    blueIsHuman: boolean,
    redTemperature: number,
    blueTemperature: number
  ) => void;
}

const RED = "oklch(0.62 0.22 25)";
const BLUE = "oklch(0.62 0.22 240)";
const RED_DIM = "oklch(0.62 0.22 25 / 0.12)";
const BLUE_DIM = "oklch(0.62 0.22 240 / 0.12)";

export default function PlayerSetup({
  defaultRedIsHuman,
  defaultBlueIsHuman,
  defaultRedTemperature,
  defaultBlueTemperature,
  onStart,
}: PlayerSetupProps) {
  const [redIsHuman, setRedIsHuman] = useState(defaultRedIsHuman);
  const [blueIsHuman, setBlueIsHuman] = useState(defaultBlueIsHuman);
  const [redTemp, setRedTemp] = useState(defaultRedTemperature);
  const [blueTemp, setBlueTemp] = useState(defaultBlueTemperature);

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "32px 20px",
      }}
    >
      <div style={{ marginBottom: 48, textAlign: "center", animation: "fadeUp 0.5s ease" }}>
        <div
          style={{
            fontSize: 52,
            fontWeight: 300,
            letterSpacing: "-0.04em",
            background:
              "linear-gradient(135deg, var(--red) 0%, oklch(0.62 0.22 130) 50%, var(--blue) 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            backgroundClip: "text",
            lineHeight: 1,
          }}
        >
          HexHex
        </div>
        <div style={{ fontSize: 13, color: "var(--muted)", marginTop: 8, letterSpacing: "0.05em" }}>
          Play Hex against our trained RL engine
        </div>
      </div>

      <div
        style={{
          display: "flex",
          gap: 16,
          width: "100%",
          maxWidth: 560,
          animation: "fadeUp 0.5s ease 0.08s both",
          flexWrap: "wrap",
        }}
      >
        <PlayerCard
          label="Red"
          subtitle="Plays top → bottom"
          color={RED}
          colorDim={RED_DIM}
          colorClass="red-range"
          isHuman={redIsHuman}
          onToggle={setRedIsHuman}
          temperature={redTemp}
          onTempChange={setRedTemp}
        />
        <PlayerCard
          label="Blue"
          subtitle="Plays left → right"
          color={BLUE}
          colorDim={BLUE_DIM}
          colorClass="blue-range"
          isHuman={blueIsHuman}
          onToggle={setBlueIsHuman}
          temperature={blueTemp}
          onTempChange={setBlueTemp}
        />
      </div>

      <div
        style={{
          marginTop: 28,
          width: "100%",
          maxWidth: 560,
          animation: "fadeUp 0.5s ease 0.16s both",
        }}
      >
        <button
          data-testid="start-game"
          onClick={() => onStart(redIsHuman, blueIsHuman, redTemp, blueTemp)}
          style={{
            width: "100%",
            padding: "14px 24px",
            border: "none",
            borderRadius: 12,
            background: "var(--text)",
            color: "var(--bg)",
            fontSize: 15,
            fontWeight: 600,
            fontFamily: "var(--font)",
            cursor: "pointer",
            letterSpacing: "-0.01em",
            transition: "opacity 0.15s, transform 0.15s",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.opacity = "0.88";
            e.currentTarget.style.transform = "translateY(-1px)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.opacity = "1";
            e.currentTarget.style.transform = "none";
          }}
        >
          Start Game
        </button>
      </div>

      <div
        style={{
          marginTop: 32,
          width: "100%",
          maxWidth: 560,
          background: "var(--card)",
          border: "1px solid var(--border)",
          borderRadius: 12,
          padding: "18px 20px",
          animation: "fadeUp 0.5s ease 0.22s both",
        }}
      >
        <div
          style={{
            fontSize: 11,
            color: "var(--muted)",
            letterSpacing: "0.06em",
            textTransform: "uppercase",
            fontWeight: 500,
            marginBottom: 10,
          }}
        >
          About this project
        </div>
        <p style={{ fontSize: 14, color: "var(--muted2)", lineHeight: 1.6, margin: 0 }}>
          A reinforcement learning engine for Hex, built by{" "}
          <span style={{ color: "var(--text)", fontWeight: 500 }}>Simon Buchholz</span>,{" "}
          <span style={{ color: "var(--text)", fontWeight: 500 }}>David Harbecke</span>, and{" "}
          <a
            href="https://cleeff.github.io/"
            target="_blank"
            rel="noreferrer"
            style={{
              color: "var(--text)",
              fontWeight: 500,
              textDecoration: "none",
              borderBottom: "1px solid var(--border2)",
            }}
          >
            Pascal Van Cleeff
          </a>
          . You are playing against a trained neural network.
        </p>
        <div style={{ marginTop: 12 }}>
          <a
            href="https://github.com/harbecke/HexHex"
            target="_blank"
            rel="noreferrer"
            style={{
              fontSize: 13,
              color: "var(--muted2)",
              textDecoration: "none",
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
              transition: "color 0.15s",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "var(--text)")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "var(--muted2)")}
          >
            <svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor" aria-hidden>
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
            </svg>
            harbecke/HexHex
          </a>
        </div>
      </div>
    </div>
  );
}

interface PlayerCardProps {
  label: string;
  subtitle: string;
  color: string;
  colorDim: string;
  colorClass: "red-range" | "blue-range";
  isHuman: boolean;
  onToggle: (v: boolean) => void;
  temperature: number;
  onTempChange: (v: number) => void;
}

function PlayerCard({
  label,
  subtitle,
  color,
  colorDim,
  colorClass,
  isHuman,
  onToggle,
  temperature,
  onTempChange,
}: PlayerCardProps) {
  return (
    <div
      data-testid={`player-card-${label.toLowerCase()}`}
      style={{
        flex: 1,
        minWidth: 240,
        background: "var(--card)",
        border: "1px solid var(--border)",
        borderRadius: 16,
        padding: "24px 24px 22px",
        display: "flex",
        flexDirection: "column",
        gap: 20,
        transition: "border-color 0.2s, box-shadow 0.2s",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <div
          style={{
            width: 36,
            height: 36,
            borderRadius: 9,
            background: colorDim,
            border: `1.5px solid ${color}55`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <div style={{ width: 14, height: 14, borderRadius: "50%", background: color }} />
        </div>
        <div>
          <div style={{ fontWeight: 600, fontSize: 15, color }}>{label}</div>
          <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 1 }}>{subtitle}</div>
        </div>
      </div>

      <TogglePill value={isHuman} onChange={onToggle} color={color} />

      <div
        style={{
          overflow: "hidden",
          maxHeight: isHuman ? 0 : 80,
          opacity: isHuman ? 0 : 1,
          transition: "max-height 0.3s ease, opacity 0.25s ease",
        }}
      >
        <TempSlider
          value={temperature}
          onChange={onTempChange}
          color={color}
          colorClass={colorClass}
        />
      </div>
    </div>
  );
}

function TogglePill({
  value,
  onChange,
  color,
}: {
  value: boolean;
  onChange: (v: boolean) => void;
  color: string;
}) {
  return (
    <div
      style={{
        display: "flex",
        background: "var(--surface)",
        borderRadius: 8,
        border: "1px solid var(--border)",
        padding: 3,
        gap: 3,
      }}
    >
      {(["Human", "AI"] as const).map((opt) => {
        const active = (opt === "Human") === value;
        return (
          <button
            key={opt}
            onClick={() => onChange(opt === "Human")}
            style={{
              flex: 1,
              border: "none",
              borderRadius: 6,
              padding: "7px 18px",
              fontSize: 13,
              fontWeight: 500,
              fontFamily: "var(--font)",
              cursor: "pointer",
              transition: "all 0.15s",
              background: active ? color : "transparent",
              color: active ? "#fff" : "var(--muted2)",
              boxShadow: active ? `0 2px 8px ${color}55` : "none",
            }}
          >
            {opt}
          </button>
        );
      })}
    </div>
  );
}

function InfoIcon({ label, tooltip }: { label: string; tooltip: string }) {
  const [hover, setHover] = useState(false);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);
  const iconRef = useRef<HTMLSpanElement | null>(null);

  function show() {
    const rect = iconRef.current?.getBoundingClientRect();
    if (rect) setPos({ left: rect.left, top: rect.bottom + 6 });
    setHover(true);
  }
  function hide() {
    setHover(false);
  }

  return (
    <>
      <span
        ref={iconRef}
        tabIndex={0}
        role="img"
        aria-label={label}
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={hide}
        style={{
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          width: 14,
          height: 14,
          borderRadius: "50%",
          border: "1px solid var(--border2)",
          color: "var(--muted)",
          fontSize: 9,
          fontWeight: 600,
          fontFamily: "var(--font)",
          cursor: "help",
          lineHeight: 1,
          userSelect: "none",
        }}
      >
        i
      </span>
      {hover &&
        pos &&
        createPortal(
          <div
            role="tooltip"
            style={{
              position: "fixed",
              left: pos.left,
              top: pos.top,
              zIndex: 1000,
              width: 260,
              maxWidth: "calc(100vw - 24px)",
              padding: "10px 12px",
              fontSize: 12,
              fontWeight: 400,
              letterSpacing: "normal",
              textTransform: "none",
              lineHeight: 1.5,
              color: "var(--text)",
              background: "var(--card)",
              border: "1px solid var(--border2)",
              borderRadius: 8,
              boxShadow: "0 4px 16px rgba(0,0,0,0.08)",
              pointerEvents: "none",
            }}
          >
            {tooltip}
          </div>,
          document.body
        )}
    </>
  );
}

function TempSlider({
  value,
  onChange,
  color,
  colorClass,
}: {
  value: number;
  onChange: (v: number) => void;
  color: string;
  colorClass: "red-range" | "blue-range";
}) {
  const pct = ((value / 2) * 100).toFixed(1) + "%";
  const style = { "--pct": pct } as CSSProperties;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 5,
            fontSize: 12,
            color: "var(--muted2)",
            letterSpacing: "0.04em",
            textTransform: "uppercase",
            fontWeight: 500,
          }}
        >
          Temperature
          <InfoIcon
            label="About temperature"
            tooltip={
              "Controls how much randomness the AI adds when choosing a move. " +
              "At 0 it always plays what it thinks is the best move, so every game " +
              "looks the same. Higher values sample from alternatives — more variety, " +
              "weaker play. Around 0.3 keeps moves strong but lets the AI pick among " +
              "near-equal options instead of repeating the same opening."
            }
          />
        </span>
        <span style={{ fontSize: 13, fontFamily: "var(--mono)", color, fontWeight: 500 }}>
          {value.toFixed(2)}
        </span>
      </div>
      <input
        type="range"
        min={0}
        max={2}
        step={0.05}
        value={value}
        className={colorClass}
        style={style}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <span style={{ fontSize: 11, color: "var(--muted)" }}>Sharp</span>
        <span style={{ fontSize: 11, color: "var(--muted)" }}>Random</span>
      </div>
    </div>
  );
}
