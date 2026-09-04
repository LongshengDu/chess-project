import type { StockfishScore } from '../../types';

// Lila's formatter is not independently importable: ceval/util.ts pulls the
// site dialog and engine graph through its runtime imports. Keep this tiny,
// source-mapped compatibility function until upstream exposes a pure module.
function renderEval(e: number): string {
  e = Math.max(Math.min(Math.round(e / 10) / 10, 99), -99);
  return (e > 0 ? '+' : '') + e.toFixed(1);
}

export function renderScore(score?: StockfishScore | null): string {
  if (score?.mate !== null && score?.mate !== undefined) return '#' + score.mate;
  return score?.cp === null || score?.cp === undefined ? '—' : renderEval(score.cp);
}
