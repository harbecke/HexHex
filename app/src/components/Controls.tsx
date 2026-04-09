interface ControlsProps {
  showRatings: boolean;
  status: "idle" | "thinking" | "gameover";
  winner: "0" | "1" | null;
  redIsHuman: boolean;
  blueIsHuman: boolean;
  agentIsBlue: boolean;
  aiSwapped: boolean;
  onToggleRatings: () => void;
  onReset: () => void;
}

export default function Controls({
  showRatings,
  status,
  winner,
  redIsHuman,
  blueIsHuman,
  aiSwapped,
  onToggleRatings,
  onReset,
}: ControlsProps) {
  let statusText = "";
  if (status === "thinking") statusText = "Agent is thinking…";
  if (status === "gameover" && winner !== null) {
    const isMixed = redIsHuman !== blueIsHuman; // exactly one human
    if (isMixed) {
      const humanIsRed = redIsHuman;
      const humanWon = (winner === "0") === humanIsRed;
      statusText = humanWon ? "You win!" : "Agent wins!";
    } else {
      statusText = winner === "0" ? "Red wins!" : "Blue wins!";
    }
  }

  return (
    <div style={{ display: "flex", gap: "1rem", alignItems: "center", flexWrap: "wrap" }}>
      <button onClick={onToggleRatings} data-testid="toggle-ratings">
        {showRatings ? "Hide Ratings" : "Show Ratings"}
      </button>
      <button onClick={onReset} data-testid="reset">
        New Game
      </button>
      {statusText && (
        <span data-testid="status" style={{ fontWeight: "bold" }}>
          {statusText}
        </span>
      )}
      {aiSwapped && (
        <span data-testid="ai-swapped" style={{ color: "#444" }}>
          AI used the swap (pie) rule.{" "}
          <a
            href="https://en.wikipedia.org/wiki/Pie_rule"
            target="_blank"
            rel="noreferrer"
          >
            What is this?
          </a>
        </span>
      )}
    </div>
  );
}
