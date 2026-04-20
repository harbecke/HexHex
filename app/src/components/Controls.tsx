import { ReactNode, useState } from "react";

interface ControlsProps {
  showRatings: boolean;
  canUndo: boolean;
  onUndo: () => void;
  onReset: () => void;
  onRestart: () => void;
  onToggleRatings: () => void;
}

export default function Controls({
  showRatings,
  canUndo,
  onUndo,
  onReset,
  onRestart,
  onToggleRatings,
}: ControlsProps) {
  return (
    <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "center" }}>
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
}: {
  onClick: () => void;
  children: ReactNode;
  active?: boolean;
  testId?: string;
  title?: string;
  hotkey?: string;
}) {
  const [hov, setHov] = useState(false);
  const border = active || hov ? "var(--border2)" : "var(--border)";
  const bg = active ? "var(--surface)" : hov ? "#ececf0" : "transparent";
  const color = active ? "var(--text)" : "var(--muted2)";
  const fullTitle = hotkey ? `${title ?? ""} (${hotkey})`.trim() : title;
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      data-testid={testId}
      title={fullTitle}
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
        cursor: "pointer",
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
