import { Player } from "../game/rules";

interface StatusBannerProps {
  status: "idle" | "thinking" | "gameover";
  winner: Player | null;
  redIsHuman: boolean;
  blueIsHuman: boolean;
  canSwap: boolean;
  aiSwapped: boolean;
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
}: StatusBannerProps) {
  const content = selectContent({ status, winner, redIsHuman, blueIsHuman, canSwap, aiSwapped });

  return (
    <div
      style={{
        minHeight: 52,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      {content}
    </div>
  );
}

function selectContent(props: StatusBannerProps) {
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
