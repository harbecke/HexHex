import { Player } from "../game/rules";
import { QualityAnnotation } from "../game/teacher";

interface StatusBannerProps {
  status: "idle" | "thinking" | "gameover";
  winner: Player | null;
  redIsHuman: boolean;
  blueIsHuman: boolean;
  canSwap: boolean;
  aiSwapped: boolean;
  teacherQuality?: QualityAnnotation | null;
  teacherLoading?: boolean;
}

/**
 * Always-rendered slot above the controls so transient messages don't push
 * the controls or the board around when they appear/disappear.
 */
export default function StatusBanner({
  status,
  winner,
  redIsHuman,
  blueIsHuman,
  canSwap,
  aiSwapped,
  teacherQuality = null,
  teacherLoading = false,
}: StatusBannerProps) {
  const primary = selectPrimary({ status, winner, redIsHuman, blueIsHuman, canSwap, aiSwapped });
  const coach = teacherQuality ? (
    <TeacherCoachLine quality={teacherQuality} />
  ) : teacherLoading ? (
    <TeacherLoadingLine />
  ) : null;

  return (
    <div
      style={{
        minHeight: 52,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 6,
      }}
    >
      {primary}
      {coach}
    </div>
  );
}

function selectPrimary(props: Omit<StatusBannerProps, "teacherQuality">) {
  const { status, winner, redIsHuman, blueIsHuman, canSwap, aiSwapped } = props;

  if (canSwap) {
    return (
      <Banner variant="neutral" testId="swap-hint">
        <strong style={{ color: "var(--text)", fontWeight: 600 }}>Pie rule:</strong>{" "}
        click the first stone to swap sides, or play your move. <PieRuleLink />
      </Banner>
    );
  }

  if (aiSwapped) {
    return (
      <Banner variant="accent" testId="ai-swapped">
        <span
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 6,
            color: "var(--text)",
            fontWeight: 600,
          }}
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 14 14"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden
          >
            <path d="M1 7h12M9 3l4 4-4 4" />
          </svg>
          The AI used the swap (pie) rule — you play Blue now.
        </span>{" "}
        <PieRuleLink />
      </Banner>
    );
  }

  if (status === "gameover" && winner !== null) {
    const isMixed = redIsHuman !== blueIsHuman;
    let msg: string;
    if (isMixed) {
      const humanIsRed = redIsHuman;
      const humanWon = (winner === "0") === humanIsRed;
      msg = humanWon ? "You win!" : "AI wins!";
    } else {
      msg = winner === "0" ? "Red wins!" : "Blue wins!";
    }
    return (
      <div
        data-testid="status"
        style={{ fontSize: 14, color: "var(--text)", fontWeight: 600, textAlign: "center" }}
      >
        {msg}
      </div>
    );
  }

  return null;
}

function TeacherCoachLine({ quality }: { quality: QualityAnnotation }) {
  const borderColor = quality.color.replace(/\)$/, " / 0.35)");
  const bgColor = quality.color.replace(/\)$/, " / 0.08)");
  return (
    <div
      data-testid="teacher-coach"
      style={{
        display: "inline-flex",
        alignItems: "baseline",
        gap: 10,
        textAlign: "center",
        lineHeight: 1.4,
        padding: "12px 20px",
        borderRadius: 12,
        border: `1px solid ${borderColor}`,
        background: bgColor,
      }}
    >
      <span style={{ color: quality.color, fontWeight: 600, fontSize: 18 }}>{quality.label}</span>
      <span style={{ color: "var(--muted)", fontFamily: "var(--mono)", fontSize: 14 }}>
        Δ {quality.delta.toFixed(2)}
      </span>
      {quality.label !== "Good" && (
        <span style={{ color: "var(--muted2)", fontSize: 14 }}>
          — better moves shown on board
        </span>
      )}
    </div>
  );
}

function TeacherLoadingLine() {
  return (
    <div
      data-testid="teacher-loading"
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 10,
        textAlign: "center",
        lineHeight: 1.4,
        padding: "12px 20px",
        borderRadius: 12,
        border: "1px dashed var(--border)",
        background: "var(--card)",
        color: "var(--muted2)",
        fontSize: 14,
      }}
    >
      <span
        aria-hidden
        style={{
          width: 8,
          height: 8,
          borderRadius: "50%",
          background: "var(--muted2)",
          animation: "pulse 1.2s ease infinite",
        }}
      />
      <span>Teacher is analyzing the position…</span>
    </div>
  );
}

function Banner({
  variant,
  testId,
  children,
}: {
  variant: "neutral" | "accent";
  testId?: string;
  children: React.ReactNode;
}) {
  const base: React.CSSProperties = {
    fontSize: 13,
    color: "var(--muted2)",
    textAlign: "center",
    maxWidth: 560,
    lineHeight: 1.6,
    padding: "10px 16px",
    borderRadius: 10,
    border: "1px solid var(--border)",
    background: "var(--card)",
  };
  const style: React.CSSProperties =
    variant === "accent"
      ? {
          ...base,
          border: "1px solid oklch(0.62 0.22 25 / 0.4)",
          background: "oklch(0.62 0.22 25 / 0.08)",
        }
      : base;
  return (
    <div data-testid={testId} style={style}>
      {children}
    </div>
  );
}

function PieRuleLink() {
  return (
    <a
      href="https://en.wikipedia.org/wiki/Pie_rule"
      target="_blank"
      rel="noreferrer"
      style={{ color: "var(--muted2)", whiteSpace: "nowrap" }}
    >
      What is this?
    </a>
  );
}
