interface ControlsProps {
  showRatings: boolean;
  temperature: number;
  canSwap: boolean;
  status: "idle" | "thinking" | "gameover";
  winner: "0" | "1" | null;
  redIsHuman: boolean;
  blueIsHuman: boolean;
  aiSwapped: boolean;
  onToggleRatings: () => void;
  onSetTemperature: (value: number) => void;
  onReset: () => void;
}

export default function Controls({
  showRatings,
  temperature,
  canSwap,
  status,
  winner,
  redIsHuman,
  blueIsHuman,
  aiSwapped,
  onToggleRatings,
  onSetTemperature,
  onReset,
}: ControlsProps) {
  const hasAI = !redIsHuman || !blueIsHuman;

  let statusText = "";
  if (status === "thinking") statusText = "AI is thinking…";
  if (status === "gameover" && winner !== null) {
    const isMixed = redIsHuman !== blueIsHuman;
    if (isMixed) {
      const humanIsRed = redIsHuman;
      const humanWon = (winner === "0") === humanIsRed;
      statusText = humanWon ? "You win!" : "AI wins!";
    } else {
      statusText = winner === "0" ? "Red wins!" : "Blue wins!";
    }
  }

  return (
    <div className="controls">
      <div className="controls-row">
        <button onClick={onReset} data-testid="reset">
          New Game
        </button>
        <button onClick={onToggleRatings} data-testid="toggle-ratings">
          {showRatings ? "Hide Ratings" : "Show Ratings"}
        </button>
        {hasAI && (
          <label className="temperature-control" data-testid="temperature-control">
            <span>AI temperature: {temperature.toFixed(2)}</span>
            <input
              type="range"
              min="0"
              max="2"
              step="0.05"
              value={temperature}
              onChange={(e) => onSetTemperature(parseFloat(e.target.value))}
              data-testid="temperature-slider"
            />
          </label>
        )}
      </div>

      <div className="controls-row controls-status">
        {statusText && (
          <span data-testid="status" className="controls-status-text">
            {statusText}
          </span>
        )}
        {canSwap && (
          <span data-testid="swap-hint" className="swap-hint">
            Pie rule: click the first stone to swap sides, or play your own move.{" "}
            <a href="https://en.wikipedia.org/wiki/Pie_rule" target="_blank" rel="noreferrer">
              What is this?
            </a>
          </span>
        )}
        {aiSwapped && (
          <span data-testid="ai-swapped" className="swap-hint">
            The AI used the swap (pie) rule.{" "}
            <a href="https://en.wikipedia.org/wiki/Pie_rule" target="_blank" rel="noreferrer">
              What is this?
            </a>
          </span>
        )}
      </div>
    </div>
  );
}
