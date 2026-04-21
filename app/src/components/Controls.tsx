import { ReactNode, useState } from "react";

interface ControlsProps {
  showRatings: boolean;
  canUndo: boolean;
  bothAI: boolean;
  paused: boolean;
  canStep: boolean;
  onUndo: () => void;
  onReset: () => void;
  onRestart: () => void;
  onToggleRatings: () => void;
  onTogglePause: () => void;
  onStep: () => void;
}

export default function Controls({
  showRatings,
  canUndo,
  bothAI,
  paused,
  canStep,
  onUndo,
  onReset,
  onRestart,
  onToggleRatings,
  onTogglePause,
  onStep,
}: ControlsProps) {
  return (
    <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "center" }}>
      {bothAI && (
        <IconBtn
          onClick={onTogglePause}
          testId="toggle-pause"
          title={paused ? "Play" : "Pause"}
          hotkey="P"
        >
          {paused ? (
            <svg
              width="13"
              height="13"
              viewBox="0 0 14 14"
              fill="currentColor"
              stroke="none"
              aria-hidden
            >
              <path d="M3 2v10l9-5z" />
            </svg>
          ) : (
            <svg
              width="13"
              height="13"
              viewBox="0 0 14 14"
              fill="currentColor"
              stroke="none"
              aria-hidden
            >
              <rect x="3" y="2" width="3" height="10" rx="0.5" />
              <rect x="8" y="2" width="3" height="10" rx="0.5" />
            </svg>
          )}
          {paused ? "Play" : "Pause"}
        </IconBtn>
      )}
      {bothAI && (
        <IconBtn
          onClick={onStep}
          testId="step"
          title="Step one AI move"
          hotkey="."
          disabled={!canStep}
        >
          <svg
            width="13"
            height="13"
            viewBox="0 0 14 14"
            fill="currentColor"
            stroke="none"
            aria-hidden
          >
            <path d="M3 2v10l7-5z" />
            <rect x="10.5" y="2" width="2" height="10" rx="0.4" />
          </svg>
          Step
        </IconBtn>
      )}
      {canUndo && (
        <IconBtn onClick={onUndo} testId="undo" title="Undo" hotkey="U">
          <svg
            width="13"
            height="13"
            viewBox="0 0 14 14"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden
          >
            <path d="M1 4h5a5 5 0 1 1 0 10" />
            <polyline points="1,1 1,4 4,4" />
          </svg>
          Undo
        </IconBtn>
      )}
      <IconBtn onClick={onReset} testId="reset" title="New Game" hotkey="N">
        <svg
          width="13"
          height="13"
          viewBox="0 0 14 14"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden
        >
          <path d="M1 7a6 6 0 1 0 1.5-4M1 3v4h4" />
        </svg>
        New Game
      </IconBtn>
      <IconBtn onClick={onRestart} testId="restart" title="Restart Game" hotkey="R">
        <svg
          width="13"
          height="13"
          viewBox="0 0 14 14"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden
        >
          <path d="M7 1v3M7 10v3M1 7h3M10 7h3" />
          <circle cx="7" cy="7" r="3" />
        </svg>
        Restart Game
      </IconBtn>
      <IconBtn
        onClick={onToggleRatings}
        active={showRatings}
        testId="toggle-ratings"
        title="Show/hide ratings"
        hotkey="S"
      >
        <svg
          width="13"
          height="13"
          viewBox="0 0 14 14"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          aria-hidden
        >
          <rect x="1" y="7" width="3" height="6" rx="1" />
          <rect x="5.5" y="4" width="3" height="9" rx="1" />
          <rect x="10" y="1" width="3" height="12" rx="1" />
        </svg>
        Ratings
      </IconBtn>
    </div>
  );
}

function IconBtn({
  onClick,
  children,
  active,
  testId,
  title,
  hotkey,
  disabled,
}: {
  onClick: () => void;
  children: ReactNode;
  active?: boolean;
  testId?: string;
  title?: string;
  hotkey?: string;
  disabled?: boolean;
}) {
  const [hov, setHov] = useState(false);
  const border = !disabled && (active || hov) ? "var(--border2)" : "var(--border)";
  const bg = disabled
    ? "transparent"
    : active
      ? "var(--surface)"
      : hov
        ? "#ececf0"
        : "transparent";
  const color = disabled ? "var(--muted)" : active ? "var(--text)" : "var(--muted2)";
  const fullTitle = hotkey ? `${title ?? ""} (${hotkey})`.trim() : title;
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      data-testid={testId}
      title={fullTitle}
      disabled={disabled}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 7,
        padding: "8px 12px 8px 14px",
        border: `1px solid ${border}`,
        borderRadius: 8,
        background: bg,
        color,
        fontSize: 13,
        fontWeight: 500,
        fontFamily: "var(--font)",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.5 : 1,
        transition: "all 0.15s",
      }}
    >
      {children}
      {hotkey && <Kbd>{hotkey}</Kbd>}
    </button>
  );
}

function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        minWidth: 18,
        height: 18,
        padding: "0 4px",
        marginLeft: 2,
        fontSize: 11,
        fontFamily: "var(--mono)",
        fontWeight: 500,
        color: "var(--muted2)",
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: 4,
        lineHeight: 1,
      }}
    >
      {children}
    </kbd>
  );
}
