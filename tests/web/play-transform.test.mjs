import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import { createRequire } from 'node:module';
const ts = createRequire(new URL('../../web/package.json', import.meta.url))('typescript');
import vm from 'node:vm';
import {transformPlaySource} from '../../web/src/adapters/play-transform.js';

const paths = ['pages/play/maia.tsx','components/Common/PlaySetupModal.tsx',
  'hooks/usePlayController/useVsMaiaController.ts','hooks/usePlayController/usePlayController.ts','components/Play/PlayControls.tsx',
  'components/Common/StatsDisplay.tsx','components/Common/ExportGame.tsx','contexts/AnalysisListContext.tsx'];
for (const path of paths) test(`upstream ${path} remains syntactically valid after local adaptation`,() => {
  const filename = fileURLToPath(new URL(`../../deps/maia-platform-frontend/src/${path}`,import.meta.url));
  const source = readFileSync(filename,'utf8').replace(/\r\n/g,'\n');
  const result = transformPlaySource(source,filename.replaceAll('\\','/'),p=>p);
  assert.equal(typeof result,'string');
  const compiled = ts.transpileModule(result,{fileName:filename,reportDiagnostics:true,
    compilerOptions:{jsx:ts.JsxEmit.ReactJSX,module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022}});
  assert.deepEqual((compiled.diagnostics || []).filter(d=>d.category===ts.DiagnosticCategory.Error),[]);
});

test('upstream clock starts after the second ply without epoch-sized move times',() => {
  const filename = fileURLToPath(new URL('../../deps/maia-platform-frontend/src/hooks/usePlayController/usePlayController.ts',import.meta.url));
  const transformed = transformPlaySource(readFileSync(filename,'utf8').replace(/\r\n/g,'\n'),filename.replaceAll('\\','/'),p=>p);
  const source = ts.createSourceFile(filename,transformed,ts.ScriptTarget.Latest,true);
  let callback;
  function visit(node) {
    if (ts.isVariableDeclaration(node) && node.name.getText(source) === 'updateClockForColor') callback = node.initializer.arguments[0].getText(source);
    ts.forEachChild(node,visit);
  }
  visit(source);
  assert.ok(callback);
  const script = ts.transpileModule(`(${callback})`,{compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText;
  let started = 0;
  const now = 1770000000000;
  const clock = (plies,time) => vm.runInNewContext(script,{Date:{now:()=>time},Math,
    moveList:Array(plies),lastMoveTime:started,whiteClock:180000,blackClock:180000,incrementSeconds:0,
    setLastMoveTime:value=>{started=value;},setWhiteClock:()=>{},setBlackClock:()=>{}});
  assert.equal(clock(0,now)('white'),0);
  assert.equal(started,0);
  assert.equal(clock(1,now+1000)('black'),0);
  assert.equal(started,now+1000);
  assert.equal(clock(2,now+1250)('white'),250);
  assert.equal(clock(3,now+1350)('black'),100);
});

test('zero-minute increment games have time available after the free opening moves',() => {
  const filename = fileURLToPath(new URL('../../deps/maia-platform-frontend/src/hooks/usePlayController/usePlayController.ts',import.meta.url));
  const transformed = transformPlaySource(readFileSync(filename,'utf8').replace(/\r\n/g,'\n'),filename.replaceAll('\\','/'),p=>p);
  const source = ts.createSourceFile(filename,transformed,ts.ScriptTarget.Latest,true);
  let initialClock;
  function visit(node) {
    if (ts.isVariableDeclaration(node) && node.name.getText(source) === 'initialClockValue') initialClock = node.initializer.getText(source);
    ts.forEachChild(node,visit);
  }
  visit(source);
  assert.equal(vm.runInNewContext(initialClock,{baseMinutes:0,incrementSeconds:1}),1000);
  assert.equal(vm.runInNewContext(initialClock,{baseMinutes:0,incrementSeconds:30}),30000);
  assert.equal(vm.runInNewContext(initialClock,{baseMinutes:3,incrementSeconds:2}),180000);
  assert.equal(vm.runInNewContext(initialClock,{baseMinutes:0,incrementSeconds:0}),0);
});

test('analysis history parses local custom games without hosted opponent fields',() => {
  const filename = fileURLToPath(new URL('../../deps/maia-platform-frontend/src/contexts/AnalysisListContext.tsx',import.meta.url));
  const transformed = transformPlaySource(readFileSync(filename,'utf8').replace(/\r\n/g,'\n'),filename.replaceAll('\\','/'),p=>p);
  const source = ts.createSourceFile(filename,transformed,ts.ScriptTarget.Latest,true);
  let parse;
  function visit(node) {
    if (ts.isVariableDeclaration(node) && node.name.getText(source) === 'parse') parse = node.initializer.getText(source);
    ts.forEachChild(node,visit);
  }
  visit(source);
  assert.ok(parse);
  const script = ts.transpileModule(`(${parse})`,{compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText;
  const parseGame = vm.runInNewContext(script);
  const custom = parseGame({game_id:'custom-1',custom_name:'Queen endgame',is_favorited:true,result:'1-0'},'custom');
  assert.equal(custom.label,'Queen endgame');
  assert.equal(custom.type,'custom');
  assert.equal(custom.is_favorited,true);
  assert.equal(custom.result,'1-0');
  assert.equal(parseGame({game_id:'empty',custom_name:''},'custom').label,'Custom Game');
  assert.equal(parseGame({game_id:'play-1',maia_name:'maia_kdd_1600',player_color:'white',result:'*'},'play').label,'You vs. Maia 1600');
  assert.equal(parseGame({game_id:'play-2',maia_name:'maia_kdd_1200',player_color:'black',result:'0-1'},'play').label,'Maia 1200 vs. You');
});

test('an unrated local save skips the upstream rating-update validation',() => {
  const filename = fileURLToPath(new URL('../../deps/maia-platform-frontend/src/hooks/usePlayController/useVsMaiaController.ts',import.meta.url));
  const transformed = transformPlaySource(readFileSync(filename,'utf8').replace(/\r\n/g,'\n'),filename.replaceAll('\\','/'),p=>p);
  const source = ts.createSourceFile(filename,transformed,ts.ScriptTarget.Latest,true);
  let guard;
  function visit(node) {
    if (ts.isIfStatement(node) && node.expression.getText(source) === 'response.player_elo != null') guard = node.getText(source);
    ts.forEachChild(node,visit);
  }
  visit(source);
  assert.ok(guard);
  const updates = [];
  for (const rating of [null,undefined,1750]) vm.runInNewContext(guard,{
    response:{player_elo:rating},safeUpdateRating:value=>updates.push(value),updateRating:()=>{},
  });
  assert.deepEqual(updates,[1750]);
});
