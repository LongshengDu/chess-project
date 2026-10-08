"""Compare saved local game timings and numerical changes against the first run."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.cache.artifacts import AnalysisStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    store = AnalysisStore()
    runs = [(path, json.loads((path/'timing.json').read_text(encoding='utf-8')),
             store.load(path/'analysis.json')) for path in args.runs]
    _, baseline, reference = runs[0]
    rows = []
    severity = {'inaccuracy', 'mistake', 'blunder'}
    for path, timing, analysis in runs:
        assert analysis['game_id'] == reference['game_id'], 'Compare the same game.'
        assert len(analysis['moves']) == len(reference['moves'])
        cp_deltas, changes = [], []
        for before, after in zip(reference['moves'], analysis['moves'], strict=True):
            assert before['fen'] == after['fen'] and before['played']['move'] == after['played']['move']
            cp_deltas.append(100*abs(after['position_eval']-before['position_eval']))
            old, new = sorted(severity.intersection(before['flags'])), sorted(severity.intersection(after['flags']))
            if old != new:
                changes.append({'ply': after['ply'], 'move': after['label'], 'before': old, 'after': new})
        phases = defaultdict(float)
        for position in timing['positions']:
            for phase in position['search']['phases']:
                phases[phase['phase']] += phase['wall_ms']/1000
        depths = [p['search']['best_depth'] for p in timing['positions']]
        rows.append({
            'run': path.name, 'timing_file': str(path/'timing.json'),
            'analysis_seconds': timing['analysis_seconds'], 'total_seconds': timing['total_seconds'],
            'speedup': baseline['analysis_seconds']/timing['analysis_seconds'],
            'runtime': timing['runtime'], 'components': timing['components'],
            'search_phase_seconds_summed': dict(phases),
            'best_move_matches': sum(a['best_move'] == b['best_move'] for a,b in
                                     zip(baseline['positions'], timing['positions'], strict=True)),
            'positions': timing['plies'], 'best_eval_delta_cp_median': statistics.median(cp_deltas),
            'best_eval_delta_cp_max': max(cp_deltas),
            'best_depth_min': min(depths), 'best_depth_median': statistics.median(depths),
            'all_legal_moves_scored': all(p['search']['coverage_complete'] for p in timing['positions']),
            'severity_changes': changes,
            'accuracy': {side: analysis['performance']['players'][side]['accuracy'] for side in ('white','black')},
            'average_accuracy': {side: analysis['accuracy_curve']['players'][side]['average_accuracy'] for side in ('white','black')},
        })
    result = {'pgn': baseline['pgn'], 'plies': baseline['plies'], 'runs': rows,
              'notes': ['Cold independent caches; unchanged depth 18 / 6-second position budgets.',
                        'Parallel search and phase times are summed workload, not elapsed game time.',
                        'One run per allocation. Evaluation differences are against the original run, not chess ground truth.',
                        'No LLM requests or tokens. This benchmark covers local analysis and accuracy measurements, not report generation.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    lines = ['# Single-game analysis performance', '', f"Game: `{baseline['pgn']}`; {baseline['plies']} plies.", '',
             '| Run | Analysis | With startup | Speedup | Maia calls | Maia time |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        c = row['components']
        lines.append(f"| {row['run']} | {row['analysis_seconds']:.2f}s | {row['total_seconds']:.2f}s | "
                     f"{row['speedup']:.2f}x | {c['maia_calls']} | {c['maia_seconds']:.2f}s |")
    lines.extend(['', '## Evaluation differences', '',
                  '| Run | Best move matches | Median / max eval difference | Best-line depth min / median | Severity changes | Accuracy White / Black |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |'])
    for row in rows:
        lines.append(f"| {row['run']} | {row['best_move_matches']}/{row['positions']} | "
                     f"{row['best_eval_delta_cp_median']:.0f} / {row['best_eval_delta_cp_max']:.0f} cp | "
                     f"{row['best_depth_min']} / {row['best_depth_median']} | {len(row['severity_changes'])} | "
                     f"{row['accuracy']['white']:.2f}% / {row['accuracy']['black']:.2f}% |")
    lines.extend(['', *result['notes'], '', 'Detailed per-position timings and search phases are in each run’s `timing.json`.',
                  'The adjacent comparison JSON includes every changed severity label and summed phase timings.', ''])
    args.output.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
