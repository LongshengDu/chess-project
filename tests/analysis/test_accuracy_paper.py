"""Publication figures preserve source evidence and exclude player identities."""
import json
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree

import chess

from analysis.accuracy.evidence import RATINGS
from analysis.game.metadata import game_metadata
from tests.analysis.render_accuracy_paper import render_example


class AccuracyPaperTests(unittest.TestCase):
    def test_both_publication_views_are_anonymous_without_modifying_source(self):
        players = {
            side: {'average_accuracy': observed, 'lichess_accuracy': observed - 2,
                   'moves_used': count, 'expected_accuracy': [expected] * len(RATINGS),
                   'absolute_deviation': [3.] * len(RATINGS)}
            for side, observed, expected, count in (('white', 95., 90., 2), ('black', 85., 88., 1))
        }
        analysis = {
            'game': game_metadata({'White': 'private-white-handle', 'Black': 'private-black-handle',
                                   'Termination': 'private-white-handle won'}),
            'headers': {'White': 'private-white-handle', 'Black': 'private-black-handle',
                        'Termination': 'private-white-handle won'},
            'start_fen': chess.STARTING_FEN,
            'accuracy_curve': {'ratings': list(RATINGS), 'expected_accuracy': [89.] * len(RATINGS),
                               'absolute_deviation': [3.] * len(RATINGS), 'players': players},
            'moves': [
                {'side': side, 'played': {'move': move}, 'accuracy': actual,
                 'maia': {str(rating): {'expected_accuracy': expected} for rating in RATINGS}}
                for side, move, actual, expected in (('white', 'e2e4', 90., 90.),
                                                     ('black', 'e7e5', 85., 88.),
                                                     ('white', 'g1f3', 100., 90.))
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'analysis.json'
            original = json.dumps(analysis).encode()
            source.write_bytes(original)
            expected = {'curve': root / 'paper' / 'curve.svg',
                        'by_move': root / 'paper' / 'by-move.svg'}
            paths = render_example(source, expected['curve'], expected['by_move'])
            self.assertEqual(paths, expected)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(set((root / 'paper').iterdir()), set(expected.values()))
            for key, path in paths.items():
                with self.subTest(view=key):
                    text = path.read_text(encoding='utf-8')
                    self.assertNotIn('private-white-handle', text)
                    self.assertNotIn('private-black-handle', text)
                    labels = ElementTree.fromstring(text).findall('.//{http://www.w3.org/2000/svg}text')
                    self.assertTrue(labels, 'Publication SVG labels must remain selectable text.')
                    self.assertIn('White — Black', [''.join(label.itertext()) for label in labels])


if __name__ == '__main__':
    unittest.main()
