import { GameTree } from '@maia/types/tree';
import { applyEngineAnalysisData } from '@maia/lib/analysis';
import { request } from './http.js';
import { cachedPositions } from './analysis-data.js';
export { request } from './http.js';
export { startGame, fetchGameMove, logGameMove, fetchPlayPlayerStats } from './play-api.js';

export const storeCustomGame = data => request('/games', data);
export const fetchMaiaGameList = (type = 'custom', page = 1) => request(`/games?type=${type}&page=${page}`);
export const updateGameMetadata = (type, id, metadata) => request(`/games/${id}`, metadata, 'PATCH');
export const deleteCustomGame = id => request(`/games/${id}`, {}, 'DELETE');
export const storeGameAnalysisCache = (id, positions) => request(`/games/${id}/analysis`, positions);
export const retrieveGameAnalysisCache = async id => {
  const cached = await request(`/games/${id}/analysis`);
  return {...cached, positions:cachedPositions(cached.positions)};
};
export const fetchWorldChampionshipGameList = async () => ({});
export const streamLichessGames = async () => {};
// The hosted opening-frequency database is separate from the Maia model.
// Without that database the upstream controller uses the actual model policy.
export const fetchOpeningBookMoves = async () => null;

export function gameFromSnapshot({game_id: id, snapshot: data}) {
  const tree = new GameTree(data.positions[0].fullFen);
  for (const [key, value] of Object.entries(data.headers || {})) tree.setHeader(key, value);
  tree.setHeader('White', data.white);
  tree.setHeader('Black', data.black);
  tree.setHeader('Result', data.result);
  let node = tree.getRoot();
  const mainline = [node];
  for (let i = 1; i < data.positions.length; i++) {
    node = tree.addMainlineNode(node, data.positions[i].fullFen, data.ucis[i - 1], data.moves[i - 1]);
    mainline.push(node);
  }
  const branches = new Map();
  for (const variation of data.variations || []) {
    const parent = variation.moves.length === 1 ? mainline[variation.ply]
      : branches.get(`${variation.ply}:${variation.moves.slice(0, -1).join(',')}`);
    if (!parent) throw new Error('Imported variation has no parent');
    const child = tree.addVariationNode(parent, variation.position.fullFen, variation.uci, variation.san);
    branches.set(`${variation.ply}:${variation.moves.join(',')}`, child);
  }
  return {id, tree, type:'custom', whitePlayer:{name:data.white}, blackPlayer:{name:data.black},
    availableMoves:[], gameType:'standard', termination:data.result === '*' ? undefined : {result:data.result, condition:'Normal',
      winner:data.result === '1-0' ? 'white' : data.result === '0-1' ? 'black' : 'none'}};
}
export const fetchAnalyzedMaiaGame = async id => {
  try {
    // Restore before mounting: the upstream node objects are mutable, so filling
    // them after mount can leave memoized panels empty until the next move.
    const [saved, cached] = await Promise.all([request(`/games/${id}`), retrieveGameAnalysisCache(id)]);
    const game = gameFromSnapshot(saved);
    applyEngineAnalysisData(game.tree, cached.positions);
    game.fullAnalysis = game.tree.fullAnalysis = cached.analysis || null;
    return game;
  } catch (error) {
    window.dispatchEvent(new CustomEvent('local-analysis-error', {detail:error.message}));
    throw error;
  }
};
export const fetchAnalyzedWorldChampionshipGame = async () => { throw new Error('Import a tournament PGN using Custom Analysis.'); };
export const fetchPgnOfLichessGame = async id => {
  const response = await fetch(`https://lichess.org/game/export/${encodeURIComponent(id)}`);
  if (!response.ok) throw new Error('Could not load the Lichess game');
  return response.text();
};
export const fetchAnalyzedPgnGame = async (id, pgn) => {
  const saved = await storeCustomGame({pgn});
  return fetchAnalyzedMaiaGame(saved.game_id);
};
