"""Profiler comparisons: compare completed runs, scores, and search depths."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import chess

from analysis.profiler.report import load_timings, summarize, write_csv


def compare(baseline, optimized):
    baseline_timings, _ = load_timings(baseline)
    optimized_timings, _ = load_timings(optimized)
    old, new = summarize(baseline_timings), summarize(optimized_timings)
    saved_old = json.loads((baseline / 'saved-analysis.json').read_text(encoding='utf-8'))['positions']
    saved_new = json.loads((optimized / 'saved-analysis.json').read_text(encoding='utf-8'))['positions']
    assert len(old['positions']) == len(new['positions']) == len(saved_old) == len(saved_new)
    rows = []
    for left, right, a, b in zip(old['positions'], new['positions'], saved_old, saved_new):
        assert chess.Board(left['fen']).fen() == chess.Board(right['fen']).fen() == chess.Board(b['fen']).fen()
        old_sf, new_sf = a['stockfish'], b['stockfish']
        assert new_sf['complete'] and len(b['maia']) == 21
        move = None
        if left['ply'] + 1 < len(saved_new):
            board = chess.Board(b['fen'])
            after = chess.Board(saved_new[left['ply'] + 1]['fen']).fen()
            for candidate in board.legal_moves:
                child = board.copy(); child.push(candidate)
                if child.fen() == after:
                    move = candidate.uci(); break
            assert move
        best_old = old_sf.get('best_move', old_sf.get('model_move'))
        best_new = new_sf.get('best_move', new_sf.get('model_move'))
        old_cp = old_sf.get('model_optimal_cp', old_sf['cp_vec'].get(best_old))
        new_cp = new_sf.get('model_optimal_cp', new_sf['cp_vec'].get(best_new))
        best_mate = best_old in old_sf['mate_vec'] or best_new in new_sf['mate_vec']
        played_mate = move in old_sf['mate_vec'] or move in new_sf['mate_vec']
        rows.append({'ply':left['ply'], 'move':left['move'],
                     'before_seconds':left['position_seconds'], 'after_seconds':right['position_seconds'],
                     'before_screen_seconds':left['screening_seconds'], 'after_screen_seconds':right['screening_seconds'],
                     'best_move_before':best_old,'best_move_after':best_new,'best_move_agrees':best_old==best_new,
                     'before_best_cp':old_cp,'after_best_cp':new_cp,
                     'best_cp_delta':None if best_mate or not move or old_cp is None or new_cp is None else abs(old_cp-new_cp),
                     'played_cp_delta':None if played_mate or not move else abs(old_sf['cp_vec'][move]-new_sf['cp_vec'][move]),
                     'best_depth':new_sf['depth'],
                     'played_depth':new_sf.get('root_move_depth_vec',{}).get(move),
                     'candidate_min_depth':new_sf.get('candidate_min_depth'),
                     'target_reached':new_sf.get('target_reached',True),
                     'coverage_complete':new_sf.get('coverage_complete',True)})
    active = [r for r in rows if r['played_depth'] is not None]
    best_deltas = [r['best_cp_delta'] for r in rows if r['best_cp_delta'] is not None]
    played_deltas = [r['played_cp_delta'] for r in rows if r['played_cp_delta'] is not None]
    return {'before_seconds':old['total_seconds'], 'after_seconds':new['total_seconds'],
            'speedup':old['total_seconds']/new['total_seconds'],
            'time_saved_percent':100*(1-new['total_seconds']/old['total_seconds']),
            'nonterminal_positions':len(active),
            'best_move_agreement':sum(r['best_move_agrees'] for r in active),
            'median_best_cp_delta':statistics.median(best_deltas) if best_deltas else None,
            'median_played_cp_delta':statistics.median(played_deltas) if played_deltas else None,
            'max_played_cp_delta':max(played_deltas) if played_deltas else None,
            'positions_with_all_candidates_at_target':sum(r['target_reached'] for r in active),
            'best_depth_range':[min(r['best_depth'] for r in active),max(r['best_depth'] for r in active)] if active else None,
            'played_depth_range':[min(r['played_depth'] for r in active),max(r['played_depth'] for r in active)] if active else None,
            'all_legal_moves_scored':all(r['coverage_complete'] for r in rows),
            'before_components':old['components_seconds'], 'after_components':new['components_seconds'],
            'before_configuration':baseline_timings['configuration'], 'after_configuration':optimized_timings['configuration'],
            'positions':rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('optimized', type=Path)
    args = parser.parse_args()
    result = compare(args.baseline, args.optimized)
    (args.optimized/'comparison.json').write_text(json.dumps(result,indent=2)+'\n', encoding='utf-8')
    write_csv(args.optimized/'comparison.csv',result['positions'])
    rows = sorted(result['positions'],key=lambda r:r['before_seconds'],reverse=True)
    lines = ['# Full-game analysis comparison','',
        f"The same game completed in **{result['after_seconds']:.3f}s**, versus **{result['before_seconds']:.3f}s**: **{result['speedup']:.2f}× speed ratio**, **{result['time_saved_percent']:.1f}% elapsed-time reduction**.",'',
        'These are individual recorded runs, not repeated controlled trials. Parallel worker totals are accumulated work and must not be treated as partitions of elapsed time.','',
        '## Recorded configuration','',
        f"Before: `{json.dumps(result['before_configuration'])}`.",'',
        f"After: `{json.dumps(result['after_configuration'])}`.",'',
        '## Slowest original positions','',
        '| Position before | Before s | After s | Best depth | Played depth | All candidates at target? |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for row in rows[:10]:
        lines.append(f"| {row['move']} | {row['before_seconds']:.3f} | {row['after_seconds']:.3f} | {row['best_depth']} | {row['played_depth']} | {row['target_reached']} |")
    lines += ['', '## Evaluation tradeoffs','',
        f"- Best move agrees on {result['best_move_agreement']} of {result['nonterminal_positions']} nonterminal positions.",
        f"- Median absolute best-score change: {result['median_best_cp_delta']} centipawns, excluding mate scores.",
        f"- Median absolute played-move score change: {result['median_played_cp_delta']} centipawns; maximum {result['max_played_cp_delta']}, excluding mate scores.",
        f"- Achieved best-line depth: {result['best_depth_range']}; played-move depth: {result['played_depth_range']}.",
        f"- All selected candidates reached their configured target in {result['positions_with_all_candidates_at_target']} played positions. All legal moves have a recorded score: {result['all_legal_moves_scored']}.",
        '- The baseline is a reference, not ground truth. Scores can change with search order, depth, time and multithreading. Use the recorded configurations above to interpret differences.', '',
        '## Sources','',
        '- [Stockfish UCI documentation](https://official-stockfish.github.io/docs/stockfish-wiki/UCI-Protocol-and-Stockfish-Commands.html): combined limits stop at the first limit; MultiPV 1 gives the best performance; threads should match available CPU cores.',
        '- [python-chess engine limits](https://python-chess.readthedocs.io/en/latest/engine.html#chess.engine.Limit): native time and depth parameters.', '',
        'Full data: [position comparison](comparison.csv), [component report](REPORT.md), [all searches](searches.csv), [saved evaluations](saved-analysis.json).','']
    (args.optimized/'COMPARISON.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='positions'},indent=2))


if __name__ == '__main__':
    main()
