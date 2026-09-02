// Source-compatible adaptation of lila/ui/lib/src/ceval/util.ts:renderEval.
import type { StockfishScore } from '../../types';

export function renderEval(e: number): string {
  e = Math.max(Math.min(Math.round(e / 10) / 10, 99), -99);
  return (e > 0 ? '+' : '') + e.toFixed(1);
}

export function renderScore(score?: StockfishScore | null): string {
  if (score?.mate !== null && score?.mate !== undefined) return '#' + score.mate;
  return score?.cp === null || score?.cp === undefined ? '—' : renderEval(score.cp);
}
