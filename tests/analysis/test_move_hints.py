"""Deterministic hint semantics, including Black scores and incomplete roots."""
import math
import json
import tempfile
import unittest
from pathlib import Path
import chess
import chess.pgn
from analysis.move_hints import LABELS, expected_score, move_flags, probability, sacrifice_hint
from analysis.game.pipeline import analyze_game
from analysis.game.summary import compact_summary
from analysis.cache.storage import write_json
from analysis.accuracy.evidence import RATINGS
from tests.coach.fixtures import FakeAnalysisSession


def sacrifice_game():
    """Frozen legal moves, independent of the user's reordered game collection."""
    with (Path(__file__).with_name('fixtures')/'declined_sacrifices.pgn').open(encoding='utf-8') as source:
        return chess.pgn.read_game(source)


def cp_for(score, side='white'):
    value = math.log(score/(1-score))/.368208
    return value if side == 'white' else -value


def ranking_for(candidates, rating):
    return [{**{key: candidate[key] for key in ('move', 'san', 'eval', 'loss')},
             'p': candidate['maia_p'][str(rating)]}
            for candidate in sorted(candidates, key=lambda c: -c['maia_p'][str(rating)])[:5]]


def row_for(before=.5, after=.5, fen=chess.STARTING_FEN, move='e2e4'):
    board = chess.Board(fen)
    side = 'white' if board.turn else 'black'
    candidates = [{'move': m.uci(), 'san': board.san(m), 'eval': cp_for(after if m.uci() == move else before, side),
                   'loss': 0, 'maia_p': {str(r): 1/board.legal_moves.count() for r in RATINGS}} for m in board.legal_moves]
    played = next(c for c in candidates if c['move'] == move)
    return {'ply': 1, 'side': side, 'fen': fen, 'position_eval': cp_for(before, side),
            'played': {k: played[k] for k in ('move', 'san', 'eval', 'loss')}, 'candidate_moves': candidates,
            'maia': {str(r): {'moves': ranking_for(candidates, r),
                              'expected_accuracy': 100., 'absolute_deviation': 0.} for r in RATINGS}}


class HintTests(unittest.TestCase):
    def test_severity_mutually_exclusive_and_black_symmetric(self):
        black = chess.Board(); black.push_uci('e2e4')
        for side, fen, move in [('white', chess.STARTING_FEN, 'e2e4'), ('black', black.fen(), 'e7e5')]:
            # Stay off inverse-logistic floating-point boundaries; Lichess uses
            # direct >= comparisons. Its blunder cutoff is 15 percentage points.
            for after, severity in [(.47, None), (.4499, 'inaccuracy'), (.43, 'inaccuracy'),
                                    (.3999, 'mistake'), (.36, 'mistake'), (.3499, 'blunder'), (.20, 'blunder')]:
                labels = move_flags(row_for(.5, after, fen, move), 1400)
                self.assertEqual(set(labels)&{'inaccuracy', 'mistake', 'blunder'}, {severity} if severity else set())
                self.assertTrue(set(labels) <= set(LABELS))
        self.assertEqual(move_flags(row_for(), 1400), [])

    def test_outcome_changes_and_opponent_error_sequence(self):
        for before, after, expected in [(.85, .2, {'missed_win','winning_to_losing'}),
                (.2, .5, {'saved_game','critical_move'}), (.5, .85, {'critical_move'}),
                (.95, .8, {'conversion_error'})]:
            self.assertTrue(expected <= set(move_flags(row_for(before, after), 1400)))
        previous = {'side':'black', 'position_eval':cp_for(.4), 'played':{'eval':cp_for(.8)}}
        self.assertIn('missed_opponent_error', move_flags(row_for(.8, .5), 1400, previous=previous))
        self.assertIn('punished_opponent_error', move_flags(row_for(.8, .8), 1400, previous=previous))
        previous['played']['eval'] = cp_for(.42)
        self.assertNotIn('punished_opponent_error', move_flags(row_for(.8, .8), 1400, previous=previous))

    def test_mate_signs_and_lost_mate_not_near_best_even_at_high_cp(self):
        for side in ('white','black'):
            for mate in ('#2','#0','#-2','#-0'):
                self.assertEqual(expected_score(mate, side), float((side=='black') == mate.startswith('#-')))
        row = row_for()
        row['played']['eval'] = '#-2'
        self.assertTrue({'allowed_mate','blunder'} <= set(move_flags(row, 1400)))
        row['position_eval'] = '#-3'
        self.assertNotIn('allowed_mate', move_flags(row, 1400))
        row['position_eval'], row['played']['eval'] = '#3', 20.
        flags = move_flags(row, 1400)
        self.assertIn('missed_mate', flags)
        self.assertNotIn('above_level_move', flags)
        row['played']['eval'] = '#0'
        self.assertIn('mate_found', move_flags(row, 1400))

    def test_only_move_requires_all_legal_roots_and_the_played_choice(self):
        row = row_for(.8, .8)
        for c in row['candidate_moves']:
            if c['move'] != 'e2e4': c['eval'] = cp_for(.45)
        complete = row['candidate_moves'][:]
        self.assertIn('only_move', move_flags(row, 1400))
        row['candidate_moves'] = [c for c in complete if c['move'] in ('e2e4','d2d4')]
        self.assertNotIn('only_move', move_flags(row, 1400))
        self.assertIn('only_move', move_flags(row, 1400, all_moves=complete))
        complete[0]['eval'] = cp_for(.8)
        self.assertNotIn('only_move', move_flags(row, 1400, all_moves=complete))

    def test_maia_trends_use_actual_level_interpolation_and_sound_moves(self):
        row = row_for(.8, .5)
        for c in row['candidate_moves']:
            for r in RATINGS:
                c['maia_p'][str(r)] = .001
        played = next(c for c in row['candidate_moves'] if c['move']=='e2e4')
        better = next(c for c in row['candidate_moves'] if c['move']=='d2d4')
        for r in RATINGS:
            played['maia_p'][str(r)] = .3-(r-1000)/10000
            better['maia_p'][str(r)] = .2+(r-1000)/2000
            row['maia'][str(r)]['moves'] = ranking_for(row['candidate_moves'], r)
        flags = move_flags(row, 1200)
        self.assertTrue({'natural_but_bad','higher_elo_improvement','stronger_maia_converges'} <= set(flags))
        self.assertAlmostEqual(probability(better, 1250), .325)
        row['played'] = {k: better[k] for k in ('move','san','eval','loss')}
        self.assertIn('above_level_move', move_flags(row, 1200))
        self.assertNotIn('above_level_move', move_flags(row, 2600))

    def test_engine_only_and_human_consensus(self):
        row = row_for(.8, .8)
        for c in row['candidate_moves']:
            c['maia_p'] = {str(r): .01 for r in RATINGS}
            if c['move'] != 'e2e4': c['eval'] = cp_for(.4)
        self.assertIn('engine_only_move', move_flags(row, 1400))
        best = next(c for c in row['candidate_moves'] if c['move']=='e2e4')
        best['maia_p'] = {str(r): .7 for r in RATINGS}
        for r in RATINGS:
            row['maia'][str(r)]['moves'] = ranking_for([best], r)
        self.assertIn('human_consensus', move_flags(row, 1400))
        self.assertNotIn('engine_only_move', move_flags(row, 1400))

    def test_accepted_sacrifice_is_sound_and_ordinary_exchange_is_not_sacrifice(self):
        row = row_for(.6,.6,'6k1/6p1/8/6B1/8/8/8/6K1 w - - 0 1','g5h6')
        reply = {'played':{'move':'g7h6'}}
        self.assertIn('sacrifice', move_flags(row,1400,reply=reply))
        row['played']['eval'] = cp_for(.1)
        self.assertNotIn('sacrifice',move_flags(row,1400,reply=reply))
        exchange = row_for(.6,.6,'6k1/6p1/7b/6B1/8/8/8/6K1 w - - 0 1','g5h6')
        self.assertNotIn('sacrifice',move_flags(exchange,1400,reply=reply))
        self.assertIn('forcing_best_move',move_flags(exchange,1400,reply=reply))

    def test_declined_bishop_and_queen_offers_and_black_symmetry(self):
        game = sacrifice_game()
        board = game.board()
        positions = {}
        for ply, move in enumerate(game.mainline_moves(), 1):
            if ply in (35, 39):
                positions[ply] = (board.copy(), move)
            board.push(move)
        self.assertEqual({ply: board.san(move) for ply, (board, move) in positions.items()},
                         {35: 'Bxg6', 39: 'Qxe4'})
        for ply, (board, move) in positions.items():
            with self.subTest(ply=ply):
                row = row_for(.6, .6, board.fen(), move.uci())
                # Historical engine scores for the frozen positions: both pass near_best.
                row['position_eval'], row['played']['eval'] = (6.92, 6.92) if ply == 35 else (21.09, 17.48)
                declined = {'played': {'move': 'e5e4' if ply == 35 else 'a5a4'}}
                self.assertIn('sacrifice', move_flags(row, 1270, reply=declined))
                self.assertIn('sacrifice', move_flags(row, 1270))  # PGN ends at the offer.
                acceptance = 'f7g6' if ply == 35 else 'f6e4'
                self.assertIn('sacrifice', move_flags(row, 1270, reply={'played': {'move': acceptance}}))
                board.push(move); board.push_uci(acceptance)
                board.push_san('Qxg6#' if ply == 35 else 'Bh7#')
                self.assertTrue(board.is_checkmate())
                mirrored = chess.Board(row['fen']).mirror()
                mirrored_move = chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square))
                black = row_for(.6, .6, mirrored.fen(), mirrored_move.uci())
                self.assertIn('sacrifice', move_flags(black, 1270))
                row['played']['eval'] = -5
                self.assertNotIn('sacrifice', move_flags(row, 1270, reply=declined))

    def test_offer_excludes_pinned_capture_full_recovery_and_old_hanging_piece(self):
        pinned = row_for(.6,.6,'4k3/4p3/8/6B1/8/8/8/4R1K1 w - - 0 1','g5f6')
        self.assertFalse(sacrifice_hint(pinned))  # ...exf6 exposes Black's king.
        exchange = row_for(.6,.6,'6k1/6p1/7b/6B1/8/8/8/6K1 w - - 0 1','g5h6')
        self.assertFalse(sacrifice_hint(exchange))
        recovered = row_for(.6,.6,'6k1/6q1/8/6B1/8/8/8/6RK w - - 0 1','g5h6')
        self.assertFalse(sacrifice_hint(recovered))  # ...Qxh6 Rxg7 recovers more material.
        hanging = row_for(.6,.6,'6k1/6p1/7B/8/8/8/8/R5K1 w - - 0 1','a1a2')
        self.assertFalse(sacrifice_hint(hanging))
        self.assertTrue(sacrifice_hint(hanging, {'played': {'move': 'g7h6'}}))

    def test_analyze_game_persists_sacrifices_for_summary_reuse(self):
        game = sacrifice_game()
        session = FakeAnalysisSession()
        self.addCleanup(session._temp.cleanup)
        analysis = analyze_game(game, session, actual_elo=1270, progress=lambda _: None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'analysis.json'
            write_json(path, analysis)
            saved = json.loads(path.read_text(encoding='utf-8'))
        summary = compact_summary(saved, 'white')
        for ply in (35,39):
            self.assertIn('sacrifice', saved['moves'][ply-1]['flags'])
            self.assertIn('sacrifice', next(row[3] for row in summary['overview'] if row[0] == ply))


if __name__ == '__main__':
    unittest.main()
