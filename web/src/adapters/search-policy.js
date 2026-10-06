// Completion means the requested search policy finished, not that every move
// reached the depth ceiling. Saved bounded results must not restart on navigation.
export function searchSatisfied(evaluation, depth, config, policy = {}, played) {
  if (!evaluation || evaluation.complete === false || !config) return false;
  // Terminal outcomes have no legal search roots or depth target. In particular,
  // a repetition draw relies on saved history and cannot be rechecked from FEN.
  if ('terminal_cp' in evaluation) return Number.isFinite(evaluation.terminal_cp);
  const candidates = [...Object.entries(policy).sort((a,b) => b[1]-a[1]).slice(0,4).map(([move]) => move), ...(played ? [played] : [])];
  if (config.strategy === 'exhaustive' && evaluation.strategy !== 'exhaustive' && !('terminal_cp' in evaluation)) return false;
  const budget = config.budgets[depth];
  const savedDepth = evaluation.target_depth ?? depth;
  if (evaluation.strategy === 'bounded' && (
      evaluation.policy_version !== config.policy_version ||
      evaluation.budget_seconds !== config.budgets[savedDepth] ||
      (evaluation.max_budget_seconds !== undefined &&
       evaluation.max_budget_seconds !== config.max_budgets?.[savedDepth]))) return false;
  if (evaluation.target_reached !== false && evaluation.depth >= depth && candidates.every(move => (evaluation.root_move_depth_vec?.[move] ?? evaluation.depth) >= depth)) return true;
  return config.strategy === 'bounded' && evaluation.strategy === 'bounded' && evaluation.complete === true &&
    evaluation.policy_version === config.policy_version && evaluation.target_depth >= depth &&
    evaluation.budget_seconds >= budget && candidates.every(move =>
      evaluation.candidate_moves?.includes(move) || (evaluation.root_move_depth_vec?.[move] ?? 0) >= depth);
}

export function searchDepth(depth, config) {
  return Object.hasOwn(config?.budgets || {}, depth) ? depth : config?.default_depth ?? depth;
}

export function depthLabel(evaluation) {
  if (!evaluation?.depth) return null;
  return evaluation.complete && evaluation.target_reached === false
    ? `d${evaluation.depth} · time limit` : `d${evaluation.depth}`;
}
