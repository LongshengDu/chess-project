import test from 'node:test';
import assert from 'node:assert/strict';
import { forEachConcurrent } from '../../web/src/adapters/concurrent.js';

const pause = ms => new Promise(resolve => setTimeout(resolve,ms));
test('positions run concurrently with a strict limit and all complete', async () => {
  let active = 0, peak = 0;
  const seen = [];
  await forEachConcurrent([0,1,2,3,4],2,async (_,index) => {
    active++; peak=Math.max(peak,active);
    await pause(index===0 ? 20 : 2);
    seen.push(index); active--;
  });
  assert.equal(peak,2); assert.equal(active,0);
  assert.notEqual(seen[0],0);
  assert.deepEqual(seen.sort(),[0,1,2,3,4]);
});
test('cancellation prevents queued positions from starting', async () => {
  let cancelled = false;
  const seen = [];
  await forEachConcurrent([0,1,2,3],2,async (_,index) => {
    seen.push(index); cancelled=true;
  },{isCancelled:() => cancelled});
  assert.deepEqual(seen,[0]);
});
test('failure stops queued work and drains already running work', async () => {
  let active = 0, stopped = 0;
  const seen = [];
  await assert.rejects(forEachConcurrent([0,1,2,3],2,async (_,index) => {
    seen.push(index); active++;
    try { await pause(index===0 ? 1 : 10); if (index===0) throw Error('failed'); }
    finally { active--; }
  },{onError:() => stopped++}),/failed/);
  assert.equal(stopped,1); assert.equal(active,0); assert.deepEqual(seen,[0,1]);
});
