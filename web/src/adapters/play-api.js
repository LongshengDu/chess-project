import { request } from './http.js';

const pendingStarts = new Map();
const pendingLogs = new Map();

export const fetchPlayConfig = () => request('/play/config');

export function startGame(playerColor, maiaVersion, gameType, sampleMoves, timeControl, _partner, startFen) {
  if (gameType !== 'play') throw new Error('This local app supports Play Maia.');
  const body = {player_color:playerColor, maia_rating:Number(maiaVersion.replace('maia_kdd_', '')),
    time_control:timeControl, sample_moves:sampleMoves, ...(startFen ? {start_fen:startFen} : {})};
  const key = JSON.stringify(body);
  if (!pendingStarts.has(key)) {
    const work = request('/play/games', {...body,request_id:crypto.randomUUID()})
      .then(data => ({gameId:data.game_id,opponentElo:data.opponent_elo}))
      .finally(() => pendingStarts.delete(key));
    pendingStarts.set(key,work);
  }
  return pendingStarts.get(key);
}

export const fetchGameMove = (moves, _maiaVersion, startFen, _piece, initialClock=0, currentClock=0, gameId) =>
  request('/play/move', {game_id:gameId,moves,...(startFen ? {start_fen:startFen} : {}),
    initial_clock:initialClock,current_clock:currentClock});

export function logGameMove(gameId, moves, moveTimes, gameOverState, _gameType, startFen, winner) {
  const body = {moves:[...moves],move_times:[...moveTimes],game_over_state:gameOverState,
    ...(startFen ? {start_fen:startFen} : {}),...(winner ? {winner} : {})};
  // A slow earlier response must not overwrite a newer position or termination.
  const work = (pendingLogs.get(gameId) || Promise.resolve()).catch(() => {})
    .then(() => request(`/play/games/${encodeURIComponent(gameId)}/moves`,body));
  pendingLogs.set(gameId,work);
  work.finally(() => { if (pendingLogs.get(gameId) === work) pendingLogs.delete(gameId); }).catch(() => {});
  return work;
}

export const fetchPlayPlayerStats = async () => {
  const stats = await request('/play/stats');
  return {playGamesPlayed:stats.play_games_played,playWon:stats.play_won,playDrawn:stats.play_drawn,
    playElo:stats.play_elo ?? null};
};

export async function savePlayedGameForAnalysis(game) {
  await logGameMove(game.id,game.tree.toMoveArray(),game.tree.toTimeArray(),
    game.termination?.type || 'not_over','play',game.tree.getRoot().fen,game.termination?.winner);
  return request(`/play/games/${encodeURIComponent(game.id)}/analysis`,{});
}
