import test from 'node:test';
import assert from 'node:assert/strict';
import { startGame,fetchGameMove,logGameMove,fetchPlayPlayerStats,savePlayedGameForAnalysis } from '../../web/src/adapters/play-api.js';

const response = data => ({ok:true,json:async () => data});

test('local starts preserve setup and deduplicate simultaneous creation', async t => {
  const calls = [];
  let finish;
  t.mock.method(globalThis,'fetch', async (url,options) => {
    calls.push({url,body:JSON.parse(options.body)});
    return new Promise(resolve => {finish = () => resolve(response({game_id:'new-game',opponent_elo:1700}));});
  });
  const args = ['black','maia_kdd_1700','play',false,'5+3',undefined,'custom starting fen'];
  const first = startGame(...args);
  const duplicate = startGame(...args);
  assert.equal(first,duplicate);
  assert.equal(calls.length,1);
  assert.equal(calls[0].url,'/api/platform/play/games');
  assert.deepEqual({...calls[0].body,request_id:undefined}, {player_color:'black',maia_rating:1700,
    sample_moves:false,time_control:'5+3',start_fen:'custom starting fen',request_id:undefined});
  assert.match(calls[0].body.request_id,/^[0-9a-f-]+$/);
  finish();
  assert.deepEqual(await first,{gameId:'new-game',opponentElo:1700});
});

test('move requests explicitly identify their local session and full history', async t => {
  let sent;
  t.mock.method(globalThis,'fetch',async (url,options) => {
    sent = {url,body:JSON.parse(options.body)};
    return response({top_move:'e7e5',move_delay:0});
  });
  assert.deepEqual(await fetchGameMove(['e2e4'],'maia_kdd_1600',null,null,180,174,'session'),{top_move:'e7e5',move_delay:0});
  assert.equal(sent.url,'/api/platform/play/move');
  assert.deepEqual(sent.body,{game_id:'session',moves:['e2e4'],initial_clock:180,current_clock:174});
});

test('full-history logs serialize and final snapshot is saved before analysis opens', async t => {
  const calls = [];
  let release;
  t.mock.method(globalThis,'fetch',async (url,options) => {
    calls.push({url,body:JSON.parse(options.body)});
    if (calls.length === 1) return new Promise(resolve => {release = () => resolve(response({player_elo:null}));});
    return response(url.endsWith('/analysis') ? {game_id:'analysis-id'} : {player_elo:null});
  });
  const first = logGameMove('serial-game',['e2e4'],[0],'not_over','play');
  const game = {id:'serial-game',tree:{toMoveArray:()=>['e2e4','e7e5'],toTimeArray:()=>[0,0],getRoot:()=>({fen:'start'})},
    termination:{type:'resign',winner:'black'}};
  const final = savePlayedGameForAnalysis(game);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(calls.length,1);
  release();
  await first;
  assert.deepEqual(await final,{game_id:'analysis-id'});
  assert.deepEqual(calls.map(call=>call.url),[
    '/api/platform/play/games/serial-game/moves',
    '/api/platform/play/games/serial-game/moves',
    '/api/platform/play/games/serial-game/analysis']);
  assert.deepEqual(calls[1].body,{moves:['e2e4','e7e5'],move_times:[0,0],game_over_state:'resign',start_fen:'start',winner:'black'});
});

test('local stats preserve unrated status and HTTP errors retain backend explanation', async t => {
  t.mock.method(globalThis,'fetch',async () => response({play_games_played:3,play_won:1,play_drawn:1,play_elo:null}));
  assert.deepEqual(await fetchPlayPlayerStats(),{playGamesPlayed:3,playWon:1,playDrawn:1,playElo:null});
  globalThis.fetch = async () => ({ok:false,status:400,json:async()=>({error:'The move is not legal.'})});
  await assert.rejects(fetchGameMove(['bad'],null,null,null,0,0,'bad-session'),/move is not legal/);
});
