export interface QualityTier {
  label: "Good" | "Inaccuracy" | "Mistake" | "Blunder";
  color: string;
}

export interface QualityAnnotation extends QualityTier {
  delta: number;
}

export interface Suggestion {
  id: number;
  score: number;
  rank: number;
}

export const TEACHER_THRESHOLDS: (QualityTier & { maxDelta: number })[] = [
  { label: "Good",       maxDelta: 0.5,      color: "oklch(0.72 0.20 145)" },
  { label: "Inaccuracy", maxDelta: 1.5,      color: "oklch(0.78 0.18 78)"  },
  { label: "Mistake",    maxDelta: 3.0,      color: "oklch(0.72 0.20 48)"  },
  { label: "Blunder",    maxDelta: Infinity, color: "oklch(0.65 0.25 15)"  },
];

function classifyDelta(delta: number): QualityAnnotation {
  for (const t of TEACHER_THRESHOLDS) {
    if (delta <= t.maxDelta) return { label: t.label, color: t.color, delta };
  }
  return { label: "Blunder", color: TEACHER_THRESHOLDS[3].color, delta };
}

export function classifyMoveQuality(
  scores: (number | null)[] | null,
  cells: (string | null)[],
  playedId: number | null
): QualityAnnotation | null {
  if (!scores || playedId === null) return null;
  const played = scores[playedId];
  if (played == null) return null;
  // `best` must be computed over moves that were *legal at decision time*:
  // currently-empty cells, plus the played cell (which was empty then).
  // The model emits a logit for every cell, so including occupied cells
  // would let a meaningless score on e.g. an opponent's stone dominate.
  let best = -Infinity;
  for (let i = 0; i < scores.length; i++) {
    if (cells[i] !== null && i !== playedId) continue;
    const s = scores[i];
    if (s != null && s > best) best = s;
  }
  if (!isFinite(best)) return null;
  return classifyDelta(best - played);
}

/** Up to `n` empty cells the AI scored strictly higher than the played move. */
export function getTopSuggestions(
  scores: (number | null)[] | null,
  cells: (string | null)[],
  playedId: number | null,
  n = 3
): Suggestion[] {
  if (!scores || playedId === null) return [];
  const played = scores[playedId];
  if (played == null) return [];

  const candidates: { id: number; score: number }[] = [];
  for (let i = 0; i < scores.length; i++) {
    if (i === playedId) continue;
    if (cells[i] !== null) continue;
    const s = scores[i];
    if (s == null) continue;
    if (s <= played) continue;
    candidates.push({ id: i, score: s });
  }
  candidates.sort((a, b) => b.score - a.score);

  return candidates.slice(0, n).map((c, idx) => ({
    id: c.id,
    score: c.score,
    rank: idx + 1,
  }));
}
