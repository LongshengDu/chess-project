import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
import config from '../../web/vite.config.mjs';
import {positionHistory} from '../../web/src/adapters/engine-history.js';

test('engine history follows the selected variation from its actual starting FEN', () => {
  const root = {fen:'custom root', parent:null};
  const main = {fen:'mainline', move:'e2e4', parent:root};
  const side = {fen:'variation', move:'d2d4', parent:root};
  const leaf = {fen:'reply', move:'d7d5', parent:side};
  assert.deepEqual(positionHistory(leaf), {start_fen:'custom root', moves:['d2d4', 'd7d5']});
  assert.deepEqual(positionHistory(main), {start_fen:'custom root', moves:['e2e4']});
  assert.deepEqual(positionHistory(root), {start_fen:'custom root', moves:[]});
  assert.throws(() => positionHistory({parent:root}), /incomplete/);
});

test('interactive upstream Maia and Stockfish calls both transmit node histories', () => {
  const filename = fileURLToPath(new URL('../../deps/maia-platform-frontend/src/hooks/useAnalysisController/useEngineAnalysis.ts', import.meta.url));
  const source = readFileSync(filename, 'utf8').replace(/\r\n/g, '\n');
  const plugin = config.plugins.find(plugin => plugin.name === 'local-analysis-adapters');
  const transformed = plugin.transform(source, filename.replaceAll('\\', '/'));
  assert.match(transformed, /MAIA_RATINGS,\s*MAIA_RATINGS,\s*positionHistory\(currentNode\)/);
  assert.match(transformed, /history: positionHistory\(currentNode\),\s*maiaCandidateMoves/);
  const ts = createRequire(new URL('../../web/package.json', import.meta.url))('typescript');
  const result = ts.transpileModule(transformed, {fileName:filename, reportDiagnostics:true,
    compilerOptions:{module:ts.ModuleKind.ESNext, target:ts.ScriptTarget.ES2022}});
  assert.deepEqual((result.diagnostics || []).filter(d => d.category === ts.DiagnosticCategory.Error), []);
});
