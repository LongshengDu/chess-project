import test from 'node:test';
import assert from 'node:assert/strict';
import { searchDepth, searchSatisfied, depthLabel } from '../../web/src/adapters/search-policy.js';

const config = {strategy:'bounded', policy_version:2, budgets:{16:1,18:3,22:6}, max_budgets:{16:10,18:30,22:60}};
const policy = {e2e4:.8,d2d4:.2};
const result = {complete:true, coverage_complete:true, strategy:'bounded', policy_version:2,
  depth:10, target_depth:28, target_reached:false, budget_seconds:6, max_budget_seconds:6,
  candidate_moves:['e2e4','d2d4','a2a3'], root_move_depth_vec:{e2e4:10,d2d4:8,a2a3:4}};

test('completed searches reuse larger requested depth and time regardless achieved depth or preset membership', () => {
  assert.equal(searchSatisfied(result,22,config,policy,'a2a3'),true);
  assert.equal(searchSatisfied(result,18,config,policy,'a2a3'),true);
  assert.equal(searchSatisfied({...result,max_budget_seconds:100},22,config,policy,'a2a3'),true);
  assert.equal(searchSatisfied({...result,target_depth:18,depth:30},22,config,policy),false);
  assert.equal(searchSatisfied({...result,budget_seconds:5,depth:30},22,config,policy),false);
  assert.equal(searchSatisfied({...result,target_depth:undefined},22,config,policy),false);
});

test('completion, policy version, strategy, and required candidates remain mandatory', () => {
  for (const update of [{complete:false},{coverage_complete:false},{policy_version:1},{strategy:'staged'}]) {
    assert.equal(searchSatisfied({...result,...update},22,config,policy,'a2a3'),false);
  }
  assert.equal(searchSatisfied(result,22,config,{g1f3:1},'a2a3'),false);
  assert.equal(searchSatisfied({...result,depth:40,root_move_depth_vec:{g1f3:40}},22,config,{g1f3:1}),false);
  assert.equal(searchSatisfied(result,22,config,{e2e4:.96,g1f3:.04}),true,'95% cutoff excludes the unused candidate');
  assert.equal(searchSatisfied(result,22,config,{d2d4:.8,e2e4:.2}),true,'candidate order is not a reuse limit');
  const four = {candidate_moves:['a2a3','b2b3','c2c3','d2d3']};
  assert.equal(searchSatisfied({...result,...four},22,config,{a2a3:.3,b2b3:.2,c2c3:.2,d2d3:.2,e2e3:.1}),true);
});

test('depth-driven strategies compare requested watchdog allowance and preserve candidate requirements', () => {
  for (const strategy of ['staged','exhaustive']) {
    const limits = {...config,strategy};
    const saved = {...result,strategy,budget_seconds:1,max_budget_seconds:60};
    assert.equal(searchSatisfied(saved,22,limits,policy,'a2a3'),true);
    assert.equal(searchSatisfied({...saved,max_budget_seconds:59},22,limits,policy),false);
    assert.equal(searchSatisfied({...saved,target_depth:21,depth:40},22,limits,policy),false);
    assert.equal(searchSatisfied(saved,22,limits,{g1f3:1}),strategy === 'exhaustive');
  }
  assert.equal(searchSatisfied({...result,strategy:'staged',candidate_moves:[]},4,
    {strategy:'staged',policy_version:2,max_budgets:{4:1}},policy),true);
});

test('finite terminal outcomes need no search roots, budget or depth', () => {
  for (const strategy of ['bounded','staged','exhaustive']) {
    assert.equal(searchSatisfied({complete:true,terminal_cp:0},22,{strategy}),true);
    assert.equal(searchSatisfied({complete:true,terminal_cp:Infinity},22,{strategy}),false);
  }
});

test('configured depth selection and achieved-depth labels retain their separate purposes', () => {
  assert.equal(searchDepth(28,{...config,default_depth:22}),22);
  assert.equal(searchDepth(16,{...config,default_depth:22}),16);
  assert.equal(depthLabel(result),'d10 · time limit');
  assert.equal(depthLabel({depth:22,complete:true,target_reached:true}),'d22');
});
