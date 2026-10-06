import test from 'node:test';
import assert from 'node:assert/strict';
import { hrefString,isLocalRoute,queryForUrl } from '../../web/src/adapters/routes.js';

test('object links retain setup choices, FEN, and false values without undefined query fields', () => {
  const href = hrefString({pathname:'/play/maia',query:{player:'black',sampleMoves:false,startFen:'8/8 b - - 0 1',missing:undefined}});
  const url = new URL(href,'http://127.0.0.1');
  assert.deepEqual(queryForUrl(url),{player:'black',sampleMoves:'false',startFen:'8/8 b - - 0 1'});
  assert.equal(isLocalRoute(href),true);
});

test('analysis catch-all IDs and hidden play session IDs have separate shapes', () => {
  assert.deepEqual(queryForUrl(new URL('http://local/analysis/example/custom')).id,['example','custom']);
  const internal = hrefString({pathname:'/play/maia',query:{id:'session',player:'white'}});
  const displayed = hrefString({pathname:'/play/maia',query:{player:'white'}});
  assert.equal(queryForUrl(new URL(internal,'http://local')).id,'session');
  assert.equal(queryForUrl(new URL(displayed,'http://local')).id,undefined);
  assert.equal(isLocalRoute('/play/maia'),true);
  assert.equal(isLocalRoute('/play/hb'),false);
  assert.equal(isLocalRoute('/analysis-other'),false);
});
