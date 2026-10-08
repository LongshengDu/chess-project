"""Game context is explicit, side-neutral and independent of source headers."""
from copy import deepcopy
import io
import unittest

import chess.pgn

from analysis.game.metadata import game_metadata, validate_game_metadata
from analysis.maia_context import native_actual_ratings


class GameMetadataTests(unittest.TestCase):
    def test_override_changes_ratings_without_editing_names_result_or_headers(self):
        headers = {'White': 'Alice', 'Black': 'Bob', 'WhiteElo': '1250', 'BlackElo': '1800',
                   'Site': 'Chess.com', 'TimeControl': '600+5', 'Result': '0-1',
                   'Termination': 'Time forfeit', 'Date': '2026.10.08', 'Event': 'Practice',
                   'Opening': 'Open game', 'ECO': 'C20', 'WhiteEloEstimate': '9999'}
        original = deepcopy(headers)
        game = chess.pgn.read_game(io.StringIO('1. e4 e5 *'))
        context = game_metadata(headers, 1600, 'lb', game=game)
        self.assertEqual(context, {
            'white': {'name': 'Alice', 'elo': 1600}, 'black': {'name': 'Bob', 'elo': 1600},
            'rating_scale': 'lichess_blitz', 'result': '0-1', 'date': '2026.10.08',
            'opening': "King's Pawn Game", 'eco': 'C20'})
        self.assertEqual(headers, original)
        self.assertEqual(native_actual_ratings({'game': context}), {'White': 1600., 'Black': 1600.})

    def test_unknown_names_and_optional_metadata_are_null(self):
        for absent in (None, '', ' ', '?', '-', '*'):
            with self.subTest(absent=absent):
                context = game_metadata({'White': absent, 'Black': absent, 'Event': absent,
                                         'Result': '*', 'Date': '????.??.??'})
                self.assertEqual(context['white'], {'name': None, 'elo': None})
                self.assertEqual(context['black'], {'name': None, 'elo': None})
                for field in ('result', 'date', 'opening', 'eco'):
                    self.assertIsNone(context[field])
                for field in ('event', 'site', 'time_control', 'termination'):
                    self.assertNotIn(field, context)
        self.assertEqual(game_metadata({'White': ' Alice '})['white']['name'], 'Alice')
        self.assertEqual(game_metadata({'Date': '2026.??.??'})['date'], '2026.??.??')

    def test_supported_scales_have_descriptive_serialized_names(self):
        for code, name in [('lb', 'lichess_blitz'), ('lr', 'lichess_rapid'),
                           ('cb', 'chess_com_blitz'), ('cr', 'chess_com_rapid')]:
            with self.subTest(scale=code):
                context = game_metadata({}, 1600, code)
                self.assertEqual(context['rating_scale'], name)
                self.assertEqual(game_metadata({}, 1600, name), context)
                validate_game_metadata(context)

    def test_opening_comes_from_eco_without_trusting_opening_header(self):
        self.assertEqual(game_metadata({'ECO': 'B20'})['opening'], 'Sicilian Defense')
        self.assertEqual(game_metadata({'ECO': 'B20', 'Opening': 'Wrong opening'})['opening'],
                         'Sicilian Defense')
        game = chess.pgn.read_game(io.StringIO('1. e4 e5 *'))
        self.assertEqual(game_metadata({'ECO': 'C20', 'Opening': 'Wrong opening'}, game=game)['opening'],
                         "King's Pawn Game")
        self.assertIsNone(game_metadata({'ECO': 'C20', 'Opening': "King's Pawn Game"})['opening'])
        for code in (None, '', '?', 'Z99'):
            with self.subTest(eco=code):
                self.assertIsNone(game_metadata({'ECO': code, 'Opening': 'Untrusted opening'})['opening'])

    def test_game_context_controls_conversion_even_when_source_headers_disagree(self):
        context = game_metadata({}, 1600, 'lb')
        analysis = {'game': context, 'headers': {'WhiteElo': '50', 'BlackElo': '4000',
                                                'Site': 'Chess.com', 'TimeControl': '?'}}
        self.assertEqual(native_actual_ratings(analysis), {'White': 1600., 'Black': 1600.})
        context['side'] = 'white'
        with self.assertRaisesRegex(ValueError, 'coaching target'):
            validate_game_metadata(context)

    def test_result_is_known_only_when_pgn_supplies_a_valid_outcome(self):
        for result in ('1-0', '0-1', '1/2-1/2'):
            self.assertEqual(game_metadata({'Result': result})['result'], result)
        for result in (None, '?', '*', 'unknown'):
            self.assertIsNone(game_metadata({'Result': result})['result'])


if __name__ == '__main__':
    unittest.main()
