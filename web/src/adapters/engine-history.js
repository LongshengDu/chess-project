// Keep the selected variation's path: a FEN alone has no repetition/history data.
export function positionHistory(node) {
  if (!node) throw new Error('A position is required for engine analysis.');
  const moves = [];
  let root = node;
  while (root.parent) {
    if (!root.move) throw new Error('The position has an incomplete move history.');
    moves.push(root.move);
    root = root.parent;
  }
  return {start_fen:root.fen, moves:moves.reverse()};
}
