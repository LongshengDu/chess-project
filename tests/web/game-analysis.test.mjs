import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';
import { GameAnalysisRun } from '../../web/src/adapters/game-analysis.js';
import { applyGamePosition, cachedPositions, terminalEvaluation } from '../../web/src/adapters/analysis-data.js';
import { request } from '../../web/src/adapters/http.js';
import viteConfig from '../../web/vite.config.mjs';
import { searchSatisfied } from '../../web/src/adapters/search-policy.js';

const ts = createRequire(new URL('../../web/package.json', import.meta.url))('typescript');
const encoder = new TextEncoder();
const streamResponse = (events, trailingNewline = true) => new Response(
  events.map(event => JSON.stringify(event)).join('\n') + (trailingNewline ? '\n' : ''),
  {headers:{'Content-Type':'application/x-ndjson'}},
);
const collect = async run => {
  const events = [];
  for await (const event of run.events()) events.push(event);
  return events;
};

test('full-game stream preserves split UTF-8, out-of-order positions and fitted ratings', async t => {
  const analysis = {positions:[], played_elo:{white:{estimate:1800}, black:{estimate:1500}}};
  const expected = [{type:'progress', message:'Maia choices…'},
    {type:'position', index:2, position:{fen:'last'}},
    {type:'position', index:0, position:{fen:'first'}}, {type:'complete', analysis}];
  const bytes = encoder.encode(expected.map(event => JSON.stringify(event)).join('\n'));
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push({url, options});
    return new Response(new ReadableStream({start(controller) {
      for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
      controller.close();
    }}));
  });
  const run = new GameAnalysisRun('saved game', 18, 'profiler-id');
  assert.deepEqual(await collect(run), expected);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, '/api/platform/games/saved%20game/analyze');
  assert.deepEqual(JSON.parse(calls[0].options.body), {run_id:run.runId, target_depth:18, profiler_id:'profiler-id'});
  assert.equal(run.finished, true);
});

test('an unfinished stream is an error and cancels the server job once', async t => {
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push({url, options});
    return url.endsWith('/cancel') ? new Response('{}') : streamResponse([{type:'progress', message:'Working'}]);
  });
  const run = new GameAnalysisRun('game', 15);
  await assert.rejects(collect(run), /ended before completion/);
  run.cancel();
  assert.equal(calls.length, 2);
  assert.equal(calls[1].url, '/api/platform/games/game/analyze/cancel');
  assert.deepEqual(JSON.parse(calls[1].options.body), {run_id:run.runId});
  assert.equal(calls[1].options.keepalive, true);
  assert.equal(run.controller.signal.aborted, true);
});

test('explicit cancellation aborts the browser request and reports the same run id', async t => {
  const calls = [];
  t.mock.method(globalThis, 'fetch', (url, options) => {
    calls.push({url, options});
    if (url.endsWith('/cancel')) return Promise.resolve(new Response('{}'));
    return new Promise((resolve, reject) => options.signal.addEventListener('abort',
      () => reject(new DOMException('Aborted', 'AbortError')), {once:true}));
  });
  const run = new GameAnalysisRun('game', 12);
  const pending = collect(run);
  run.cancel();
  await assert.rejects(pending, {name:'AbortError'});
  assert.equal(calls.length, 2);
  assert.equal(JSON.parse(calls[0].options.body).run_id, JSON.parse(calls[1].options.body).run_id);
});

test('server errors surface their message and server cancellation is terminal', async t => {
  let events = [{type:'error', message:'No complete legal move scores'}];
  const calls = [];
  t.mock.method(globalThis, 'fetch', async url => {
    calls.push(url);
    return url.endsWith('/cancel') ? new Response('{}') : streamResponse(events, false);
  });
  await assert.rejects(collect(new GameAnalysisRun('game', 18)), /No complete legal move scores/);
  events = [{type:'cancelled'}];
  const before = calls.length;
  assert.deepEqual(await collect(new GameAnalysisRun('game', 18)), events);
  assert.equal(calls.length-before, 1);
});

const position = (fen, ply) => ({fen, ply,
  maia:{maia_kdd_1600:{policy:{e2e4:.2, d2d4:.8}, value:.5}},
  stockfish:{depth:12, cp_vec:{e2e4:10, d2d4:20}, mate_vec:{}, complete:true},
});
function node(fen) {
  return {fen, analysis:{stockfish:{depth:30}},
    addMaiaAnalysis(value) { this.analysis.maia = value; },
    addStockfishAnalysis(value) { this.analysis.stockfish = value; }};
}

test('tree updates use original indices, sorted policies and authoritative common scores', () => {
  const nodes = [node('first'), node('last')];
  assert.equal(applyGamePosition(nodes, 1, position('last', 1), 'maia_kdd_1600', raw=>raw), nodes[1]);
  assert.equal(nodes[0].analysis.stockfish.depth, 30);
  assert.equal(nodes[1].analysis.stockfish.depth, 12);
  assert.deepEqual(Object.keys(nodes[1].analysis.maia.maia_kdd_1600.policy), ['d2d4','e2e4']);
  assert.throws(() => applyGamePosition(nodes, 0, position('different', 0), 'maia_kdd_1600', raw=>raw), /does not match/);
});

test('terminal cache adaptation does not damage the canonical saved analysis', () => {
  const terminal = {fen:'mate', ply:1, maia:{}, stockfish:{terminal_cp:10000, cp_vec:{}, mate_vec:{}}};
  const complete = [position('first', 0), terminal];
  const cached = cachedPositions(complete);
  assert.equal(cached[1].stockfish.terminal_cp, 10000);
  assert.equal(complete[1].stockfish.terminal_cp, 10000);
  assert.deepEqual(Object.keys(complete[0].maia.maia_kdd_1600.policy), ['e2e4','d2d4']);
  const board = node('mate');
  applyGamePosition([board], 0, terminal, 'maia_kdd_1600', raw=>raw);
  assert.equal(board.analysis.stockfish.terminal_cp, 10000);
  const empty = {...terminal, stockfish:{cp_vec:{}}};
  assert.equal(cachedPositions([empty])[0].stockfish, undefined);
  applyGamePosition([board], 0, empty, 'maia_kdd_1600', () => assert.fail('No root to reconstruct'));
});

test('upstream cache restoration preserves mate and history-only draw without a new search', () => {
  const filename = fileURLToPath(new URL('../../deps/maia-platform-frontend/src/lib/analysis.ts', import.meta.url));
  const source = readFileSync(filename, 'utf8').replace(/\r\n/g, '\n');
  const adapter = viteConfig.plugins.find(plugin=>plugin.name === 'local-analysis-adapters');
  const transformed = adapter.transform(source, filename.replaceAll('\\', '/'));
  const parsed = ts.createSourceFile(filename, transformed, ts.ScriptTarget.Latest, true);
  const declaration = parsed.statements.filter(ts.isVariableStatement)
    .flatMap(statement=>statement.declarationList.declarations)
    .find(item=>item.name.getText(parsed) === 'applyEngineAnalysisData');
  assert.ok(declaration);
  const compiled = ts.transpileModule(`(${declaration.initializer.getText(parsed)})`,
    {compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText;
  const restore = vm.runInNewContext(compiled, {terminalEvaluation,
    reconstructCachedStockfishAnalysis:()=>assert.fail('Terminal evidence must not reconstruct a legal-root search')});
  const startFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
  for (const [cp, checkmate] of [[0, false], [10000, true]]) {
    const board = node(startFen);
    board.analysis = {};
    const position = {ply:0, fen:startFen, stockfish:{depth:0, cp_vec:{}, mate_vec:checkmate ? {'':0} : {},
      terminal_cp:cp, is_checkmate:checkmate, complete:true, result:checkmate ? '1-0' : '1/2-1/2'}};
    restore({getMainLine:()=>[board]}, cachedPositions([position]));
    assert.equal(board.analysis.stockfish.model_optimal_cp, cp);
    assert.equal(board.analysis.stockfish.model_move, '');
    assert.equal(board.analysis.stockfish.is_checkmate, checkmate);
    assert.equal(board.analysis.stockfish.result, position.stockfish.result);
    assert.equal(board.analysis.stockfish.sent, true);
    for (const strategy of ['bounded','staged','exhaustive']) {
      assert.equal(searchSatisfied(board.analysis.stockfish, 18, {strategy}, {}, null), true);
    }
  }
});

function hookHarness(nodes, {gameId='game', profiling=false} = {}) {
  const states = [], moved = [], errors = [], batches = [];
  let updates = 0;
  const tree = {getMainLine:()=>nodes, toPGN:()=> '[White "A"]\n\n1. e4 *'};
  const maia = {status:'ready'};
  const stockfish = {profiling, isReady:()=>true, beginBatch:owner=>batches.push(['start',owner]),
    endBatch:owner=>batches.push(['end',owner])};
  const react = {useContext:value=>value, useRef:current=>({current}), useCallback:fn=>fn,
    useEffect:fn=>fn(), useState:initial=> {
      const index = states.length;
      states.push(initial);
      return [initial, value=>{states[index] = typeof value === 'function' ? value(states[index]) : value;}];
    }};
  const dependencies = {'react':react, 'react-hot-toast':{toast:{error:message=>errors.push(message)}},
    './engines':{MaiaEngineContext:maia, StockfishEngineContext:stockfish, stockfishEvaluation:raw=>raw},
    './api':{request, storeCustomGame:data=>request('/games', data)},
    './game-analysis':{GameAnalysisRun}, './analysis-data':{applyGamePosition}};
  const source = readFileSync(new URL('../../web/src/adapters/deep-analysis.js', import.meta.url), 'utf8');
  const compiled = ts.transpileModule(source, {compilerOptions:{module:ts.ModuleKind.CommonJS,
    target:ts.ScriptTarget.ES2022}}).outputText;
  const exports = {};
  vm.runInNewContext(compiled, {exports, require:name=>dependencies[name], performance, Set, Error});
  const hook = exports.useDeepAnalysis({gameTree:tree, gameId, currentMaiaModel:'maia_kdd_1600',
    setCurrentNode:value=>moved.push(value), setAnalysisState:fn=>{updates=fn(updates);}});
  return {hook, states, moved, errors, batches, tree, get updates() { return updates; }};
}

test('hook consumes one full-game job, retains ratings and never posts partial cache data', async t => {
  const positions = [position('first', 0), position('last', 1)];
  const analysis = {positions, played_elo:{white:{estimate:1750}, black:{estimate:1650}}};
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push({url, body:JSON.parse(options.body)});
    return streamResponse([{type:'position', index:1, position:positions[1]},
      {type:'position', index:0, position:positions[0]}, {type:'complete', analysis}]);
  });
  const view = hookHarness([node('first'), node('last')]);
  await view.hook.startAnalysis(18);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, '/api/platform/games/game/analyze');
  assert.equal(view.errors.length, 0);
  assert.equal(view.states[1].isComplete, true);
  assert.equal(view.states[1].currentMoveIndex, 2);
  assert.deepEqual(view.tree.fullAnalysis, analysis);
  assert.deepEqual(view.moved.map(value=>value.fen), ['last','first']);
  assert.equal(view.updates, 3);
  assert.deepEqual(view.batches.map(([action])=>action), ['start','end']);
});

test('unsaved trees create a PGN once and completed-cache streams need no position callbacks', async t => {
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push({url, body:JSON.parse(options.body)});
    return url.endsWith('/games') ? new Response(JSON.stringify({game_id:'created'}))
      : streamResponse([{type:'complete', analysis:{positions:[position('first',0)], played_elo:{}}}]);
  });
  const view = hookHarness([node('first')], {gameId:null});
  await view.hook.startAnalysis(12);
  await view.hook.startAnalysis(12);
  assert.equal(calls.filter(call=>call.url.endsWith('/games')).length, 1);
  assert.equal(calls[0].body.pgn, view.tree.toPGN());
  assert.equal(calls[1].url, '/api/platform/games/created/analyze');
  assert.equal(view.states[1].isComplete, true);
  assert.equal(view.errors.length, 0);
});

test('profiler hooks send the profiler id without duplicating server position timings', async t => {
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push({url, body:JSON.parse(options.body)});
    if (url.endsWith('/profiler/start')) return new Response(JSON.stringify({run_id:'profiler'}));
    if (url.endsWith('/profiler/finish')) return new Response('{}');
    return streamResponse([{type:'complete', analysis:{positions:[position('first',0)]}}]);
  });
  const view = hookHarness([node('first')], {profiling:true});
  await view.hook.startAnalysis(15);
  assert.deepEqual(calls.map(call=>call.url), ['/api/platform/profiler/start',
    '/api/platform/games/game/analyze', '/api/platform/profiler/finish']);
  assert.equal(calls[1].body.profiler_id, 'profiler');
  assert.equal(view.errors.length, 0);
});

test('client validation failure cleans a profiler run even after the server completes', async t => {
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push({url, body:JSON.parse(options.body)});
    if (url.endsWith('/profiler/start')) return new Response(JSON.stringify({run_id:'profiler'}));
    if (url.endsWith('/profiler/finish')) return new Response('{}');
    return streamResponse([{type:'complete', analysis:{positions:[]}}]);
  });
  const view = hookHarness([node('first')], {profiling:true});
  await view.hook.startAnalysis(18);
  assert.equal(view.errors.length, 1);
  assert.deepEqual(calls.at(-1), {url:'/api/platform/profiler/finish', body:{run_id:'profiler', cancelled:true}});
});

test('a rejected profiler finish gets one best-effort cancellation cleanup', async t => {
  const finishes = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    if (url.endsWith('/profiler/start')) return new Response(JSON.stringify({run_id:'profiler'}));
    if (url.endsWith('/profiler/finish')) {
      finishes.push(JSON.parse(options.body));
      return finishes.length === 1 ? new Response(JSON.stringify({error:'Could not finish timing'}), {status:500}) : new Response('{}');
    }
    return streamResponse([{type:'complete', analysis:{positions:[position('first',0)]}}]);
  });
  const view = hookHarness([node('first')], {profiling:true});
  await view.hook.startAnalysis(18);
  assert.equal(finishes.length, 2);
  assert.deepEqual(finishes[1], {run_id:'profiler', cancelled:true});
  assert.equal(view.errors.length, 1);
});

test('interactive autosave remains adapted separately from full-game persistence', () => {
  const config = readFileSync(new URL('../../web/vite.config.mjs', import.meta.url), 'utf8');
  assert.match(config, /'hooks\/useAnalysisController\/useAutoSave': 'analysis-save\.js'/);
});

test('autosave keeps interactive and cancelled work but skips the completed server result', async () => {
  const slots = [], effects = [], posts = [];
  let cursor = 0, batch = false, holdSave = false, releaseSave;
  const tree = {positions:[position('first', 0)]};
  const stockfish = {isBatchRunning:()=>batch};
  const equal = (left, right) => left?.length === right.length && right.every((value, i)=>value === left[i]);
  const react = {
    useContext:value=>value,
    useMemo(fn, deps) {
      const index = cursor++;
      if (!slots[index] || !equal(slots[index].deps, deps)) slots[index] = {deps, value:fn()};
      return slots[index].value;
    },
    useCallback(fn, deps) { return this.useMemo(()=>fn, deps); },
    useRef(initial) {
      const index = cursor++;
      if (!slots[index]) slots[index] = {current:initial};
      return slots[index];
    },
    useState(initial) {
      const index = cursor++;
      if (!slots[index]) slots[index] = {value:initial};
      return [slots[index].value, value=>{slots[index].value=value;}];
    },
    useEffect(fn, deps) {
      const index = cursor++;
      if (!slots[index] || !equal(slots[index].deps, deps)) effects.push(()=> {
        slots[index]?.cleanup?.();
        slots[index] = {deps, cleanup:fn()};
      });
    },
  };
  react.useCallback = react.useCallback.bind(react);
  const dependencies = {'react':react, './engines':{StockfishEngineContext:stockfish},
    '@maia/lib/analysis':{collectEngineAnalysisData:game=>game.positions, generateAnalysisCacheKey:JSON.stringify},
    './api':{storeGameAnalysisCache:async (id, positions)=> {
      posts.push({id, positions});
      if (holdSave) await new Promise(resolve=>{releaseSave=resolve;});
    }}};
  const source = readFileSync(new URL('../../web/src/adapters/analysis-save.js', import.meta.url), 'utf8');
  const compiled = ts.transpileModule(source, {compilerOptions:{module:ts.ModuleKind.CommonJS,
    target:ts.ScriptTarget.ES2022}}).outputText;
  const exports = {};
  vm.runInNewContext(compiled, {exports, require:name=>dependencies[name], console,
    setInterval:()=>1, clearInterval:()=>{}});
  const render = analysisState => {
    cursor = 0;
    const hook = exports.useAutoSave({game:{id:'game', type:'custom'}, gameTree:tree,
      enableAutoSave:true, analysisState});
    while (effects.length) effects.shift()();
    return hook;
  };
  render(0);
  tree.positions = [position('first', 0), position('next', 1)];
  await render(1).saveAnalysis();
  assert.equal(posts.length, 1, 'interactive work is saved');
  batch = true;
  tree.positions = [position('first', 0), position('next', 1), position('last', 2)];
  await render(2).saveAnalysis();
  assert.equal(posts.length, 1, 'no autosave during the server job');
  tree.fullAnalysis = {positions:tree.positions, played_elo:{white:{estimate:1700}}};
  batch = false;
  await render(3).saveAnalysis();
  assert.equal(posts.length, 1, 'the server already persisted the complete result');
  batch = true;
  tree.positions = [...tree.positions, position('partial', 3)];
  render(4);
  batch = false;
  await render(4).saveAnalysis();
  assert.equal(posts.length, 2, 'cancelled position work remains saveable');
  tree.positions = [...tree.positions, position('interactive', 4)];
  holdSave = true;
  const pending = render(5).saveAnalysis();
  batch = true;
  tree.positions = [...tree.positions, position('complete', 5)];
  render(6);
  tree.fullAnalysis = {positions:tree.positions, played_elo:{white:{estimate:1750}}};
  batch = false;
  render(7);
  releaseSave();
  await pending;
  await render(7).saveAnalysis();
  assert.equal(posts.length, 3, 'a stale save completion cannot dirty the newer canonical result');
});
