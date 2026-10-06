import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { createRequire } from 'node:module';
const ts = createRequire(new URL('../../web/package.json', import.meta.url))('typescript');
import { simplifyHeader } from '../../web/src/adapters/header-transform.js';

const source = readFileSync(new URL('../../deps/maia-platform-frontend/src/components/Common/Header.tsx', import.meta.url), 'utf8');

test('keeps upstream desktop/mobile layouts with only local Play Maia and Analysis navigation', () => {
  const transformed = simplifyHeader(source);
  assert.equal((transformed.match(/startGame\('againstMaia'\)/g) || []).length, 2);
  assert.equal((transformed.match(/href="\/analysis"/g) || []).length, 2);
  assert.equal((transformed.match(/src="\/maia-ios-icon\.png"/g) || []).length, 3);
  assert.match(transformed, /const desktopLayout =/);
  assert.match(transformed, /const mobileLayout =/);
  assert.match(transformed, /className="flex w-\[90%\] flex-row items-center justify-between"/);
  assert.match(transformed, /data-component="app-header"/);
  assert.doesNotMatch(transformed, /handAndBrain|lichess\.org|Discord|Leaderboard|connectLichess|logout|href="\/(?:puzzles|drills|turing|broadcast|leaderboard|blog|profile|settings)"/);
  const compiled = ts.transpileModule(transformed, {
    fileName: 'Header.tsx', reportDiagnostics: true,
    compilerOptions: { jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
  });
  assert.deepEqual(compiled.diagnostics.filter(entry => entry.category === ts.DiagnosticCategory.Error), []);
});

test('desktop and mobile Play Maia buttons open setup directly and close the mobile menu', () => {
  const transformed = simplifyHeader(source);
  assert.equal((transformed.match(/onClick=\{\(\) => startGame\('againstMaia'\)\}/g) || []).length, 2);
  assert.equal((transformed.match(/>\s+Play Maia\s+<\/button>/g) || []).length, 2);
  assert.doesNotMatch(transformed, /showPlayDropdown|setShowPlayDropdown|arrow_drop_down|aria-haspopup|AnimatePresence|>PLAY</);
  assert.match(transformed, /const handleRouteChange = \(\) => \{\s+setShowMenu\(false\)/);
  assert.match(transformed, /setShowMenu\(false\)\s+setPlaySetupModalProps\(\{ playType: playType \}\)/);
});

test('fails visibly when pinned upstream anchors change or become ambiguous', () => {
  assert.throws(() => simplifyHeader(source.replace('  const userInfo = ', '  const renamedUserInfo = ')), /needs review/);
  assert.throws(() => simplifyHeader(source + '\n  const userInfo = duplicate'), /needs review/);
  assert.equal(simplifyHeader(source.replace(/\r?\n/g, '\r\n')), simplifyHeader(source));
});
