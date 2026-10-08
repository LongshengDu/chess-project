// Completion means the requested search policy finished, not that every move
// reached the depth ceiling. Saved bounded results must not restart on navigation.
export function searchSatisfied(evaluation, depth, config, policy = {}, played) {
  if (!evaluation || evaluation.complete !== true || !config) return false;
  // Terminal outcomes have no legal search roots or depth target. In particular,
  // a repetition draw relies on saved history and cannot be rechecked from FEN.
  if ('terminal_cp' in evaluation) return Number.isFinite(evaluation.terminal_cp);
  if (evaluation.coverage_complete !== true || evaluation.strategy !== config.strategy ||
      evaluation.policy_version !== config.policy_version) return false;
  const budget = config.strategy === 'bounded' ? config.budgets?.[depth] : config.max_budgets?.[depth];
  const savedBudget = config.strategy === 'bounded' ? evaluation.budget_seconds : evaluation.max_budget_seconds;
  if (!Number.isFinite(budget) || !Number.isFinite(savedBudget) ||
      !Number.isInteger(evaluation.target_depth) || evaluation.target_depth < depth || savedBudget < budget) return false;
  if (config.strategy === 'exhaustive' || (config.strategy === 'staged' && depth <= 4)) return true;
  // Match the upstream request's 95%-mass cutoff and the backend's top-four cap.
  const candidates = played ? [played] : [];
  let cumulative = 0, count = 0;
  for (const [move, probability] of Object.entries(policy).filter(([,p]) => Number.isFinite(p) && p > 0).sort((a,b) => b[1]-a[1])) {
    candidates.push(move);
    cumulative += probability;
    if (++count === 4 || cumulative >= .95) break;
  }
  return candidates.every(move => evaluation.candidate_moves?.includes(move));
}

export function searchDepth(depth, config) {
  return Object.hasOwn(config?.budgets || {}, depth) ? depth : config?.default_depth ?? depth;
}

export function depthLabel(evaluation) {
  if (!evaluation?.depth) return null;
  return evaluation.complete && evaluation.target_reached === false
    ? `d${evaluation.depth} · time limit` : `d${evaluation.depth}`;
}
