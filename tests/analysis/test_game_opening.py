"""Local Lichess names follow ECO and played positions, including transpositions."""
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import chess.pgn

from analysis.game import opening


class OpeningTests(unittest.TestCase):
    def test_same_eco_distinguishes_actual_opening_and_latest_variation(self):
        cases = [
            ('A00', '1. Nh3', 'Amar Opening'),
            ('A00', '1. a3', "Anderssen's Opening"),
            ('C20', '1. e4 e5', "King's Pawn Game"),
            ('B01', '1. e4 d5 2. exd5 Qxd5 3. Nc3 Qa5', 'Scandinavian Defense: Main Line'),
        ]
        for eco, moves, expected in cases:
            with self.subTest(eco=eco, moves=moves):
                game = chess.pgn.read_game(StringIO(moves + ' *'))
                self.assertEqual(opening.opening_name(eco, game), expected)

    def test_position_match_recognizes_a_different_move_order(self):
        game = chess.pgn.read_game(StringIO('1. Nc3 d5 2. e4 dxe4 3. Nxe4 e5 *'))
        self.assertEqual(opening.opening_name('B01', game), 'Van Geet Opening: Grünfeld Defense')

    def test_does_not_guess_from_unrelated_eco_or_ambiguous_code(self):
        game = chess.pgn.read_game(StringIO('1. d4 d5 *'))
        self.assertIsNone(opening.opening_name('B20', game))
        self.assertIsNone(opening.opening_name('A00'))
        for code in (None, '?', 'Z99', 'B20/anything', 20):
            self.assertIsNone(opening.opening_name(code, game))

    def test_common_code_name_is_returned_only_when_unambiguous(self):
        self.assertEqual(opening.opening_name('B20'), 'Sicilian Defense')

    def test_missing_dependency_has_actionable_error_and_no_header_fallback(self):
        opening._volume.cache_clear()
        self.addCleanup(opening._volume.cache_clear)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(opening, 'OPENINGS_DIR', Path(directory)):
            with self.assertRaisesRegex(FileNotFoundError, 'submodule update --init deps/lichess-openings'):
                opening.opening_name('B20')


if __name__ == '__main__':
    unittest.main()
