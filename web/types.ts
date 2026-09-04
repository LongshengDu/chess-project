import type { Key, Role } from '@lichess-org/chessground/types';

export type Color = 'white' | 'black';
export type MoveSound = 'move' | 'capture' | 'check' | 'checkmate';
export type MaterialSide = { pieces: Role[]; score: number };
export type Material = Record<Color, MaterialSide>;

export type PositionState = {
  fen: string;
  turn: Color;
  lastMove: Key[] | null;
  check: boolean;
  material: Material;
  moveSounds: MoveSound[];
};

export type RoundState = PositionState & {
  modelName: string;
  modelElo: number;
  gameOver: boolean;
  result: string | null;
  status: string;
  moves: string[];
  history: PositionState[];
  dests: Record<string, Key[]>;
  promotions: Record<string, string[]>;
  canTakeback: boolean;
  canClaimDraw: boolean;
  canResign: boolean;
  sounds: MoveSound[];
  pgn: string;
};

export type RoundPrefs = {
  blindfold: boolean;
  coordinates: boolean;
  sound: boolean;
};

export type PromotionRole = 'q' | 'n' | 'r' | 'b';

export type AnalysisGame = {
  title: string;
  subtitle: string;
  white: string;
  black: string;
  result: string;
  moves: string[];
  ucis: string[];
  positions: AnalysisPositionState[];
};

export type AnalysisPositionState = PositionState & {
  ply: number;
  dests: Record<string, Key[]>;
  promotions: Record<string, string[]>;
  gameOver: boolean;
  status: string;
};

export type AnalysisVariationStep = {
  uci: string;
  san: string;
  position: AnalysisPositionState;
};

export type StockfishScore = {
  cp: number | null;
  mate: number | null;
  label: string;
};

export type AnalysisMove = {
  uci: string;
  san: string;
  probability: number;
  rank: number;
  played: boolean;
  stockfish: StockfishScore | null;
};

export type PositionAnalysis = {
  ply: number;
  turn: Color;
  elo: number;
  legalMoveCount: number;
  moves: AnalysisMove[];
  otherProbability: number;
  stockfish: StockfishScore | null;
  stockfishDepth: number | null;
  playedMove: string | null;
};
