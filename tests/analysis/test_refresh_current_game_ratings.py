"""The explicit rerun includes new filenames and never accepts a cached fit."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tests.analysis import refresh_current_game_ratings as runner
from tests.analysis.test_player_rating_shared_curve_affine import evidence_fixture


class CurrentGameRatingRefreshTests(unittest.TestCase):
    def test_discovers_all_pgn_names_naturally_without_fixed_game_count(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ('game24.pgn', 'game2.pgn', 'game10.PGN', 'another.pgn', 'notes.txt'):
                (directory/name).touch()
            (directory/'nested.pgn').mkdir()
            self.assertEqual([path.name for path in runner.discover_games(directory)],
                             ['another.pgn', 'game2.pgn', 'game10.PGN', 'game24.pgn'])

    def test_empty_folder_does_not_claim_a_successful_rerun(self):
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, 'No PGN games'):
                runner.discover_games(temporary)

    def test_current_signature_is_recomputed_and_evidence_is_not_changed(self):
        evidence = evidence_fixture()
        before = deepcopy(evidence)
        analysis = {'headers': {'Site': 'Chess.com', 'TimeControl': '600',
                                'WhiteElo': '1450', 'BlackElo': '1500'}}
        case = {'analysis': analysis, 'evidence': evidence, 'evidence_key': 'a'*64,
                'directory': Path('unused-output')}
        with patch.object(runner, 'fit_evidence', wraps=runner.fit_evidence) as fit, \
             patch.object(runner, 'export_figures') as export:
            first = runner.refit_case(case)
            signature = deepcopy(analysis['rating_fit'])
            second = runner.refit_case(case)
        self.assertEqual(fit.call_count, 2)
        self.assertEqual(export.call_count, 2)
        self.assertEqual(signature, analysis['rating_fit'])
        self.assertEqual(first, second)
        self.assertEqual(before, evidence)
        self.assertEqual(analysis['headers']['WhiteElo'], '1450')
        self.assertEqual(analysis['played_elo_method'], 'shared_curve_affine')
        self.assertEqual(analysis['played_elo_scale']['scale'], 'cr')


if __name__ == '__main__':
    unittest.main()
