"""Balanced investigation selection and a compact view of saved game evidence."""
import chess
import numpy as np
from analysis.position_evaluation import RATINGS
from analysis.move_hints import expected_score, mate_for
from analysis.maia_context import native_player_rating
from analysis.accuracy.comparison import AccuracyComparison


def decision_rows(analysis, side):
    if side not in ('white', 'black'):
        raise ValueError('Coaching side must be white or black.')
    return [row for row in analysis['moves'] if row['side'] == side
            and chess.Board(row['fen']).legal_moves.count() > 1]


def candidates_for_investigation(rows, native_levels):
    moments = []
    for index, row in enumerate(rows):
        level = native_levels[row['side']]
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


def compact_summary(analysis, side):
    """Prepare a targeted request from shared evidence without mutating it."""
    decisions = decision_rows(analysis, side)
    selected_side = side
    curve = analysis.get('accuracy_curve')
    native_levels = {side: (native_player_rating(analysis, side) or 1500)
                     for side in ('white', 'black')}
    moments = candidates_for_investigation(analysis['moves'], native_levels)
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
        if row['side'] == selected_side and index+1 < len(analysis['moves']):
            reply = analysis['moves'][index+1]
            level = native_levels[reply['side']]
            rating = min((int(r) for r in reply['maia'] if r.isdigit()), key=lambda r: (abs(r-level), r))
            choices = reply['maia'][str(rating)]['moves']
            if choices:
                top = choices[0]
                likely_reply = [rating, top['san'], top['p'], top['eval']]
        overview.append([row['ply'], row['label'], row['played']['loss'], row.get('flags', []), likely_reply])
    return {
        'game': {**analysis['game'], 'maia_rating_scale': 'lichess_blitz',
                 **{side: {**analysis['game'][side], 'maia_elo': level} for side, level in native_levels.items()}},
        'coaching': {**analysis['coaching'], 'side': selected_side},
        'performance': analysis.get('performance'),
        'position_difficulty': (AccuracyComparison(analysis).compare_positions(ratings=[
            min(RATINGS, key=lambda r: (abs(r-native_levels[selected_side]), r))]) if curve else None),
        'accuracy_curve': ({
            **{key: curve[key] for key in
               ('rating_scale', 'rating_name', 'ratings', 'expected_accuracy', 'absolute_deviation',
                'position_selection', 'pooling')},
            'players': {side: {key: player.get(key) for key in
                ('average_accuracy', 'lichess_accuracy', 'moves_used')}
                for side, player in curve['players'].items()}
        } if curve else None),
        'stages_reached': stages, 'required_decisions': min(3, len(decisions)), 'critical_moments': chosen,
        'overview_columns': ['ply', 'move', 'loss', 'flags', 'likely_reply'],
        'likely_reply_columns': ['maia_rating', 'san', 'p', 'eval'], 'likely_reply_conditioning': 'equal_rating',
        'overview': overview}
