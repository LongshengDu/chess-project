import assert from 'node:assert/strict';
import { searchDepth, searchSatisfied, depthLabel } from '../../web/src/adapters/search-policy.js';

const config = {strategy:'bounded', policy_version:1, budgets:{12:1,15:3,18:6}};
const policy = {e2e4:.8,d2d4:.2};
const result = {complete:true, strategy:'bounded', policy_version:1, depth:16, target_depth:18,
  target_reached:false, budget_seconds:6, candidate_moves:['e2e4','d2d4','a2a3'],
  root_move_depth_vec:{e2e4:16,d2d4:14,a2a3:12}};
assert.equal(searchSatisfied(result,18,config,policy,'a2a3'),true);
assert.equal(searchSatisfied(result,15,config,policy,'a2a3'),true);
assert.equal(searchSatisfied(result,18,{...config,budgets:{18:12}},policy,'a2a3'),false);
assert.equal(searchSatisfied({...result, max_budget_seconds:30},18,{...config,max_budgets:{18:20}},policy,'a2a3'),false);
assert.equal(searchSatisfied({...result, depth:18, target_reached:true},18,{...config,budgets:{18:12}},policy),false);
assert.equal(searchDepth(18,{budgets:{12:2,15:5},default_depth:15}),15);
assert.equal(searchDepth(12,{budgets:{12:2,15:5},default_depth:15}),12);
assert.equal(searchSatisfied(result,18,{...config,policy_version:2},policy,'a2a3'),false);
assert.equal(searchSatisfied({...result,complete:false},18,config,policy,'a2a3'),false);
assert.equal(searchSatisfied(result,18,config,{g1f3:1},'a2a3'),false);
assert.equal(searchSatisfied(result,18,{...config,strategy:'staged'},policy,'a2a3'),false);
assert.equal(searchSatisfied({...result,depth:18},18,{...config,strategy:'staged'},policy,'a2a3'),false);
assert.equal(searchSatisfied({depth:18,complete:true,strategy:'staged',root_move_depth_vec:{e2e4:18,d2d4:18,a2a3:18}},18,config,policy,'a2a3'),true);
assert.equal(searchSatisfied(result,12,{...config,strategy:'exhaustive'},policy,'a2a3'),false);
assert.equal(depthLabel(result),'d16 · time limit');
assert.equal(depthLabel({depth:18,complete:true,target_reached:true}),'d18');
console.log('Search policy: bounded completion, cache reuse, upgrades, coverage, and depth labels passed.');
