// Adapt canonical server evidence to the pinned upstream mutable game tree.
export function terminalEvaluation(raw) {
  return {...raw, sent:true, model_move:'', model_optimal_cp:raw.terminal_cp,
    cp_relative_vec:{}, winrate_vec:{}, winrate_loss_vec:{}};
}

export function normalizePosition(position) {
  const result = {...position};
  if (position.maia) result.maia = Object.fromEntries(
    Object.entries(position.maia).map(([model, evaluation]) => [model, {
      ...evaluation,
      policy:Object.fromEntries(Object.entries(evaluation.policy).sort((a, b) => b[1] - a[1])),
    }]),
  );
  return result;
}

export function cachedPositions(positions) {
  return positions.map(position => {
    const result = normalizePosition(position);
    // The upstream adapter handles explicit terminal results without inventing
    // a legal root. Only an empty, nonterminal search is unusable cached data.
    if (result.stockfish && !('terminal_cp' in result.stockfish) &&
        !Object.keys(result.stockfish.cp_vec || {}).length) delete result.stockfish;
    return result;
  });
}

export function applyGamePosition(nodes, index, position, activeModel, stockfishEvaluation) {
  const node = nodes[index];
  if (!Number.isInteger(index) || !node || node.fen !== position.fen) {
    throw new Error('Saved analysis does not match this game position. Reload the game and retry.');
  }
  const normalized = normalizePosition(position);
  const stockfish = normalized.stockfish;
  if (normalized.maia) node.addMaiaAnalysis(normalized.maia, activeModel);
  if (stockfish && ('terminal_cp' in stockfish || Object.keys(stockfish.cp_vec || {}).length)) {
    // A completed shared-pipeline result is authoritative even when an earlier
    // interactive search used a different depth or configuration.
    delete node.analysis.stockfish;
    node.addStockfishAnalysis(stockfishEvaluation(stockfish, node.fen), activeModel);
  }
  return node;
}
