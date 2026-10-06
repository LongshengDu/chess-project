"""Offline selected-estimator comparison and nullable-interval exports."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import chess
import numpy as np

from analysis.player_rating.parameters import RATINGS
from tests.analysis import compare_shared_curve_games as comparison


class RatingComparisonTests(unittest.TestCase):
    def test_selected_point_only_estimator_uses_context_and_exports_no_interval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            games, output = root/'games', root/'comparison'
            games.mkdir()
            (games/'game0.pgn').write_text('[WhiteElo "1600"]\n[BlackElo "1700"]\n'
                                           '[WhiteEloEstimate "1800"]\n[BlackEloEstimate "1750"]\n'
                                           '\n1. e4 e5 *\n', encoding='utf-8')
            analysis_path = games/'output/game0-full/analysis.json'
            analysis_path.parent.mkdir(parents=True)
            saved = {'headers': {'WhiteElo': '1600', 'BlackElo': '1700'},
                     'moves': [{'side': 'white', 'position_eval': .2}, {'side': 'black', 'position_eval': .3}],
                     'played_elo': {'white': {'estimate': 1500}, 'black': {'estimate': 1550}},
                     'played_elo_method': 'bayesian_shared_curve'}
            analysis_path.write_text(json.dumps(saved), encoding='utf-8')
            evidence = {side: {'conditioning': 'equal_opponent', 'rating_grid': list(RATINGS),
                              'observations': []} for side in ('White', 'Black')}
            board = chess.Board()
            for side, move in (('White', 'e2e4'), ('Black', 'e7e5')):
                legal = sorted(candidate.uci() for candidate in board.legal_moves)
                quality = np.linspace(40., 100., len(legal))
                policies = []
                for rating in range(len(RATINGS)):
                    probabilities = np.exp((quality-quality.max())/(45.-rating))
                    policies.append((probabilities/probabilities.sum()).tolist())
                evidence[side]['observations'].append({
                    'played_index': legal.index(move), 'qualities': {'root': quality.tolist(), 'position': quality.tolist()},
                    'maia_probabilities': policies, 'weight': 1.})
                board.push_uci(move)
            original_evidence = deepcopy(evidence)
            provenance = {'source': 'synthetic cached evidence', 'source_sha256': 'fixture'}
            with patch.dict(comparison.CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='shared_curve_affine'), \
                    patch.object(comparison, 'load_evidence', return_value=(evidence, 'a'*64, provenance)), \
                    patch.object(comparison, 'fit_evidence', wraps=comparison.fit_evidence) as fit, \
                    patch.object(comparison, 'refresh_performance') as performance, \
                    patch.object(comparison, 'export_figures') as export, patch('sys.stdout', new=io.StringIO()):
                result = comparison.run(games, output)
            # This deliberately minimal fixture covers fit/export dispatch;
            # full legal replay/performance has its own pipeline tests.
            performance.assert_called_once()
            self.assertEqual(evidence, original_evidence)
            contextual = fit.call_args.args[0]
            self.assertEqual(contextual['White']['actual_rating'], 1600)
            self.assertEqual(contextual['Black']['actual_rating'], 1700)
            self.assertIn('position_win_probability', contextual['White']['observations'][0])
            self.assertEqual(result['method_id'], 'shared_curve_affine')
            self.assertEqual(result['method_name'], 'Shared-curve affine fit')
            self.assertEqual(result['summary']['selected_rating']['players'], 2)
            for row in result['players']:
                self.assertIsNone(row['interval_low'])
                self.assertIsNone(row['interval_high'])
                self.assertIsInstance(row['selected_rating'], int)
                self.assertEqual(row['method_id'], result['method_id'])
            fitted = export.call_args.args[0]
            self.assertIsNone(fitted['central_interval'])
            self.assertIsNone(fitted['players']['White']['interval'])
            refreshed = json.loads(analysis_path.read_text(encoding='utf-8'))
            self.assertIsNone(refreshed['played_elo_central_interval'])
            self.assertEqual(refreshed['played_elo_method'], 'shared_curve_affine')
            with (output/'comparison.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(all(row['interval_low'] == row['interval_high'] == '' for row in rows))
            svg = (output/'commercial-comparison.svg').read_text(encoding='utf-8')
            self.assertIn('Shared-curve affine fit (shared_curve_affine)', svg)
            self.assertEqual(json.loads((output/'original-analysis/game0.json').read_text(encoding='utf-8')), saved)

    def test_default_output_preserves_historical_bayesian_artifacts(self):
        self.assertEqual(comparison.OUTPUT.name, 'selected-rating-production')

    def test_no_reference_comparisons_produce_explicit_empty_metrics(self):
        result = comparison.metrics([{'selected_rating': None, 'reference': 1800}], 'selected_rating')
        self.assertEqual(result['players'], 0)
        self.assertIsNone(result['mean_absolute_error'])
        self.assertIsNone(result['maximum_absolute_error'])
        json.dumps(result, allow_nan=False)

    def test_legacy_comparison_figures_keep_the_old_method_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            comparison.comparison_figure([{'game': 'game0', 'side': 'White', 'reference': 1800,
                                            'current_shared_curve': 1700, 'previous_saved': 1600}], Path(temporary))
            svg = (Path(temporary)/'commercial-comparison.svg').read_text(encoding='utf-8')
            self.assertIn('Current shared curve', svg)
            self.assertNotIn('shared_curve_affine', svg)


if __name__ == '__main__':
    unittest.main()
