"""Compact model evidence; the complete engine results remain on disk."""
from __future__ import annotations

import json

from analysis.game.context import leadup_context
from analysis.game.summary import compact_summary
from analysis.game.history import history_at
from analysis.move_hints import expected_score, probability
from analysis.cache import write_json
from analysis.player_rating.scale import native_player_rating


def rounded(value):
    if isinstance(value, float):
        return float(f'{value:.5g}')
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    return value


def baseline(row, actual_elo, include=()):
    """Reduce choices around actual_elo expressed on Maia's Lichess Blitz scale."""
    available = sorted(map(int, row['maia']))
    ratings = sorted({min(available, key=lambda r: abs(r-target))
                      for target in (actual_elo, actual_elo+200, actual_elo+400, actual_elo+600)})
    maia = {str(r): row['maia'][str(r)][:3] for r in ratings}
    roots = {row['played']['move'], *include} | {m['move'] for entries in maia.values() for m in entries}
    return {**{k: row[k] for k in ('ply', 'label', 'side', 'stage', 'position_eval', 'played')},
        'flags': row.get('flags', []),
        'maia': maia, 'candidate_moves': [{**{k: c[k] for k in ('move', 'san', 'eval', 'loss')},
            'maia_p': {str(r): c['maia_p'][str(r)] for r in ratings}}
            for c in row['candidate_moves'] if c['move'] in roots]}


def compact_evidence(result, actual_elo):
    def visit(value):
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        if value.get('conditioning') == 'equal_rating' and 'covered_probability' in value:
            ratings = list(value['covered_probability'])
            return {**{k: v for k, v in value.items() if k not in ('moves', 'covered_probability', 'engine_reply_p')},
                'ratings': list(map(int, ratings)),
                'moves': [{**{k: v for k, v in move.items() if k != 'maia_p'},
                           'p': [move['maia_p'][r] for r in ratings]} for move in value['moves']],
                'covered_probability': [value['covered_probability'][r] for r in ratings],
                'engine_reply_p': [value['engine_reply_p'][r] for r in ratings]}
        if 'candidate_moves' in value and 'position_eval' in value:
            return baseline(value, actual_elo)
        if value.get('baseline') and value.get('candidate', {}).get('stockfish'):
            return {key: (baseline(item, actual_elo, [value['candidate']['stockfish']['move']])
                          if key == 'baseline' else visit(item)) for key, item in value.items()}
        # Engine-defense continuations are secondary checks. Send their leading
        # choice at each rating; preserve full human-reply curves and all local
        # results. More continuation choices remain available through tools.
        return {key: ({r: moves[:1] for r, moves in item.items()} if key == 'maia_after_defense' and item
                      else visit(item)) for key, item in value.items()}
    return rounded(visit(result))


class ToolEvidence(dict):
    """Serialize chess observations as compact JSON for Codex."""
    def __str__(self):
        return json.dumps(self, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def prepare_initial_evidence(library):
    """Prepare bounded checked comparisons locally before the first model call."""
    overview = compact_summary(library.analysis)
    evidence = []
    library.phase = 'initial'
    try:
        for moment in overview['critical_moments']:
            row = library.analysis['moves'][moment['ply']-1]
            side = row['side']
            level = (native_player_rating(library.analysis, side)
                     or native_player_rating(library.analysis, side, fitted=True) or 1500)
            alternatives = [c for c in row['candidate_moves'] if c['move'] != row['played']['move']]
            if alternatives:
                # Choose a plausible human comparison, not a preordained
                # engine improvement. Practical merit is investigated below.
                viable = [c for c in alternatives if expected_score(c['eval'], side) >= .45] or alternatives
                candidate = max(viable, key=lambda c: (sum(probability(c, min(2600, level+s))
                                    for s in (0, 200, 400, 600)), c['move']))
                result = library.call('compare_played_vs_candidate', {'ply': row['ply'], 'candidate': candidate['move']})
            else:
                # Forced moves are normally excluded by decision_rows.
                result = library.call('explore_candidate', {'ply': row['ply'], 'candidate': row['played']['move']})
                result['baseline'] = compact_evidence(row, level)
                result['diagram'] = library.diagrams.render(history_at(library.analysis, row['ply']))
            evidence.append(result)
        if not evidence:
            evidence.append(library.call('get_position', {'ply': 1}))
    finally:
        library.phase = 'followup'
    # Share preceding moves and snapshots across overlapping moments. These
    # are read-only game facts, not additional model/engine investigations.
    context = leadup_context(library.analysis, [m['ply'] for m in overview['critical_moments']] or [1])
    package = ToolEvidence(**overview, leadup_context=context, initial_evidence=evidence)
    write_json(library.directory / 'initial_evidence.json', package)
    return str(package)
