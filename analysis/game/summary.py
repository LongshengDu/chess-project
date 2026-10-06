"""Balanced investigation selection and a compact view of saved game evidence."""
import chess
import numpy as np
from analysis.position_evaluation import RATINGS, header_elo
from analysis.move_hints import expected_score, mate_for
from analysis.player_rating.scale import analysis_scale, native_player_rating


def decision_rows(analysis):
    return [row for row in analysis['moves'] if row['side'] == analysis['selected_player']['side']
            and chess.Board(row['fen']).legal_moves.count() > 1]


def candidates_for_investigation(rows, selected_side, actual_elo, fitted, headers, *, native_levels=None):
    moments = []
    for index, row in enumerate(rows):
        level = (native_levels[row['side']] if native_levels is not None else
                 actual_elo if row['side'] == selected_side else
                 header_elo(headers, row['side']) or fitted[row['side']]['estimate'] or 1500)
        near = min(RATINGS, key=lambda r: abs(r-level))
        stronger = sorted({min(RATINGS, key=lambda r: abs(r-level-step)) for step in (200, 400, 600)})
        played = next(c for c in row['candidate_moves'] if c['move'] == row['played']['move'])
        p_base = played['maia_p'][str(near)]
        p_high = float(np.mean([played['maia_p'][str(r)] for r in stronger]))
        loss = played['loss']
        before, after = (expected_score(value, row['side']) for value in (row['position_eval'], played['eval']))
        score_loss = max(0., before-after)
        reasons, priority = [], 0.
        priority += sum({'blunder': 5, 'mistake': 3, 'inaccuracy': 1, 'sacrifice': 3,
            'only_move': 3, 'saved_game': 3, 'higher_elo_improvement': 2, 'above_level_move': 3,
            'missed_opponent_error': 2}.get(flag, 0) for flag in row.get('flags', []))
        if priority:
            reasons.append('move_flags')
        # Raw pawn differences in already decided positions can dwarf the
        # game's actual teaching moments. Match the hint score-loss scale.
        if score_loss >= .05:
            reasons.append('objective_loss'); priority += min(score_loss*20, 8)
        if isinstance(row['position_eval'], str) or isinstance(played['eval'], str):
            changed_mate = any(mate_for(row['position_eval'], side) != mate_for(played['eval'], side)
                               for side in ('white', 'black'))
            reasons.append('verify_mating_line'); priority += 3 if changed_mate else 1
        if p_base > .10 and p_high < p_base*.45:
            reasons.append('played_move_fades_at_higher_ratings'); priority += 2
        alternatives = []
        for candidate in row['candidate_moves']:
            support = float(np.mean([candidate['maia_p'][str(r)] for r in stronger]))
            if candidate['move'] != played['move'] and support >= .15 and candidate['loss'] is not None and candidate['loss'] <= .35:
                alternatives.append({'move': candidate['move'], 'san': candidate['san'], 'p_higher': round(support, 4)})
        if alternatives and score_loss >= .05:
            reasons.append('attainable_human_alternative'); priority += 2
        if loss is not None and loss <= .25 and (p_high > p_base+.08 or p_base < .15 and p_high > .15):
            reasons.append('strong_decision_above_actual_level'); priority += 3
        if chess.Board(row['fen']).is_check() and loss is not None and loss <= .25:
            reasons.append('accurate_defense'); priority += 1
        previous = rows[index-1] if index else None
        if previous and expected_score(previous['position_eval'], previous['side']) - \
                expected_score(previous['played']['eval'], previous['side']) >= .10:
            reasons.append('response_to_opponent_error'); priority += 1
        if reasons:
            moments.append({'ply': row['ply'], 'label': row['label'], 'side': row['side'], 'stage': row['stage'],
                'priority': round(priority, 2), 'reasons': reasons, 'loss': loss,
                'alternatives': sorted(alternatives, key=lambda c: -c['p_higher'])[:2]})
    return sorted(moments, key=lambda m: (-m['priority'], m['ply']))


def rating_note(analysis):
    """Describe the saved estimator accurately in the agent's compact input."""
    name = analysis.get('played_elo_name') or analysis.get('played_elo_method', 'Saved rating estimate').replace('_', ' ').title()
    central_interval = analysis.get('played_elo_central_interval', analysis.get('played_elo_confidence'))
    if central_interval is not None:
        interval = f'central {central_interval:.0%} interval'
    elif any(player.get('interval') is not None for player in analysis.get('played_elo', {}).values()):
        interval = 'saved interval'
    else:
        interval = 'point estimates; no uncertainty interval'
    return ' '.join(part for part in (
        f'{name}; {interval}.',
        f"Displayed played-level scale: {analysis.get('played_elo_scale', {}).get('name', 'Lichess Blitz')}. Maia evidence and tool ratings use Lichess Blitz.",
        analysis.get('played_elo_description'), analysis.get('played_elo_interval_scope')) if part)


def compact_summary(analysis):
    """Read saved analysis without calculating hints or mutating its records."""
    selected = analysis['selected_player']
    native_levels = {side: (native_player_rating(analysis, side)
                            or native_player_rating(analysis, side, fitted=True) or 1500)
                     for side in ('white', 'black')}
    moments = candidates_for_investigation(analysis['moves'], selected['side'], selected['actual_elo'],
                                           analysis['played_elo'], analysis['headers'], native_levels=native_levels)
    decisions = decision_rows(analysis)
    decision_plies = {r['ply'] for r in decisions}
    relevant = [m for m in moments if m['ply'] in decision_plies]
    # Seed distinct parts of the game rather than three adjacent moves in one
    # tactical sequence. Soft proximity discount: a truly major nearby moment
    # can still win. No move names, fixed game plies or mandatory hint categories.
    chosen = []
    remaining = relevant.copy()
    while remaining and len(chosen) < 3:
        candidate = max(remaining, key=lambda m: (
            m['priority'] / (1 + sum(max(0., 1-abs(m['ply']-c['ply'])/8) for c in chosen)),
            -m['ply']))
        chosen.append(candidate)
        remaining.remove(candidate)
    stages = list(dict.fromkeys(row['stage'] for row in decisions))
    for phase in stages:
        if not any(m['stage'] == phase for m in chosen):
            candidate = next((m for m in relevant if m['stage'] == phase), None)
            if candidate:
                chosen.append(candidate)
            else:
                row = next(r for r in decisions if r['stage'] == phase)
                chosen.append({'ply': row['ply'], 'label': row['label'], 'stage': phase})
    for row in decisions:
        if len(chosen) >= min(3, len(decisions)):
            break
        if row['ply'] not in {m['ply'] for m in chosen}:
            chosen.append({'ply': row['ply'], 'label': row['label'], 'stage': row['stage']})
    # Include all decisions in very short games, even when no heuristic flags them.
    if len(decisions) <= 3:
        chosen = [{'ply': row['ply'], 'label': row['label'], 'stage': row['stage'],
                   'played': row['played']} for row in decisions]
    chosen = [{**{k: analysis['moves'][m['ply']-1][k] for k in ('ply', 'label', 'stage')},
               'flags': analysis['moves'][m['ply']-1].get('flags', [])} for m in chosen]
    chosen.sort(key=lambda moment: moment['ply'])
    # Give the agent a cheap whole-game human overview, including moments not
    # selected for deep preparation. Read existing next-position Maia/eval data;
    # do not run engines, recalculate flags or assume the actual reply was best.
    overview = []
    for index, row in enumerate(analysis['moves']):
        likely_reply = None
        if row['side'] == selected['side'] and index+1 < len(analysis['moves']):
            reply = analysis['moves'][index+1]
            level = native_levels[reply['side']]
            rating = min(map(int, reply['maia']), key=lambda r: (abs(r-level), r))
            choices = reply['maia'][str(rating)]
            if choices:
                top = choices[0]
                likely_reply = [rating, top['san'], top['p'], top['eval']]
        overview.append([row['ply'], row['label'], row['played']['loss'], row.get('flags', []), likely_reply])
    return {'headers': {k: v for k, v in analysis['headers'].items() if k in ('White', 'Black', 'WhiteElo', 'BlackElo', 'Result', 'Site', 'TimeControl')},
        'selected_player': selected, 'played_elo': analysis['played_elo'],
        'actual_rating_scale': analysis_scale(analysis),
        'played_elo_scale': {key: value for key, value in analysis.get('played_elo_scale',
            {'scale': 'lb', 'name': 'Lichess Blitz'}).items() if key in ('scale', 'name', 'native_scale')},
        'maia_rating_scale': 'Lichess Blitz',
        'selected_player_maia_elo': native_levels.get(selected['side']),
        'performance': analysis.get('performance'),
        'elo_note': rating_note(analysis),
        'stages_reached': stages, 'required_decisions': min(3, len(decisions)), 'critical_moments': chosen,
        'overview_columns': ['ply', 'move', 'loss', 'flags', 'likely_reply'],
        'likely_reply_columns': ['maia_rating', 'san', 'p', 'eval'], 'likely_reply_conditioning': 'equal_rating',
        'overview': overview}
