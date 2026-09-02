import type { AnalysisGame, AnalysisVariationStep, PositionAnalysis, RoundState } from './types';

async function requestJson<T extends object>(path: string, body?: object): Promise<T> {
  const response = await fetch(
    path,
    body && {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    },
  );
  const data: T | { error: string } = await response.json().catch(() => ({
    error: response.statusText,
  }));
  if (!response.ok) throw new Error('error' in data ? data.error : response.statusText);
  return data as T;
}

export const roundApi = {
  state: () => requestJson<RoundState>('/api/state'),
  move: (uci: string) => requestJson<RoundState>('/api/move', { uci }),
  reply: () => requestJson<RoundState>('/api/reply', {}),
  action: (name: string) => requestJson<RoundState>('/api/action', { name }),
  newGame: () => requestJson<RoundState>('/api/new-game', {}),
};

export const analysisApi = {
  importPgn: (pgn: string) => requestJson<AnalysisGame>('/api/analysis/pgn', { pgn }),
  position: (ply: number, moves: string[] = []) =>
    requestJson<PositionAnalysis>('/api/analysis/position', { ply, moves }),
  move: (ply: number, moves: string[], uci: string) =>
    requestJson<AnalysisVariationStep>('/api/analysis/move', { ply, moves, uci }),
};
