"""Per-move expectations are saved during analysis and graphed without rescoring."""
import copy
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from statistics import mean
from unittest.mock import patch
from xml.etree import ElementTree

import chess
import chess.pgn

from analysis.accuracy.by_move import expected_accuracy_by_move, export_move_curve, main
from analysis.accuracy.evidence import RATINGS
from analysis.accuracy.figures import DEFAULT_FIGURE_NAMES, export_saved_figures
from analysis.accuracy.service import refresh_saved_curve
from analysis.game.pipeline import analyze_game
from analysis.game.metadata import game_metadata
from analysis.cache.artifacts import AnalysisStore
from analysis.accuracy.lichess import move_accuracy, win_percent
from tests.coach.fixtures import FakeAnalysisSession


FORCED_GAP_PGN = ('[FEN "8/8/8/8/7r/2k5/8/K7 w - - 0 41"]\n\n'
                  '41. Ka2 Ra4+ 42. Kb1 Rh4 43. Kc1 *')


class AccuracyCurveByMoveTests(unittest.TestCase):
    def analysis(self, pgn='1. e4 e5 2. Nf3 Nc6 *'):
        game = chess.pgn.read_game(io.StringIO(pgn))
        self.assertFalse(game.errors)
        session = FakeAnalysisSession(game.board().fen())
        self.addCleanup(session._temp.cleanup)
        return analyze_game(game, session, progress=lambda _: None)

    def weighted_analysis(self):
        analysis = self.analysis()
        for index, played_probability in ((0, .25), (2, .75)):
            position = analysis['positions'][index]
            played = analysis['moves'][index]['played']['move']
            position['stockfish']['cp_vec']['a2a3'] = -1000
            for rating, probability in ((1600, played_probability), (2600, .9)):
                policy = position['maia'][f'maia_kdd_{rating}']['policy']
                policy.update(dict.fromkeys(policy, 0.))
                policy.update({played: probability, 'a2a3': 1-probability})
        refresh_saved_curve(analysis)
        return analysis

    def test_each_point_is_probability_weighted_and_not_cumulative(self):
        analysis = self.weighted_analysis()
        before = copy.deepcopy(analysis)
        result = expected_accuracy_by_move(analysis)
        first, second = result['players']['white']
        first_bad = move_accuracy(win_percent(15), win_percent(-1000))
        second_bad = move_accuracy(win_percent(20), win_percent(-1000))
        self.assertAlmostEqual(first['expected_accuracy'], .25*100 + .75*first_bad)
        self.assertAlmostEqual(second['expected_accuracy'], .75*100 + .25*second_bad)
        self.assertAlmostEqual(analysis['moves'][0]['maia']['1600']['absolute_deviation'],
                               .25*abs(100-first['expected_accuracy'])
                               + .75*abs(first_bad-first['expected_accuracy']))
        self.assertNotAlmostEqual(second['expected_accuracy'],
                                  mean((first['expected_accuracy'], .75*100 + .25*second_bad)))
        self.assertEqual(result['maia_elo'], 1600)
        self.assertEqual([point['played'] for point in result['players']['white']], ['e2e4', 'g1f3'])
        self.assertEqual(analysis, before)

    def test_requested_native_anchor_selects_its_saved_expectation(self):
        result = expected_accuracy_by_move(self.weighted_analysis(), maia_elo=2600)
        first_bad = move_accuracy(win_percent(15), win_percent(-1000))
        self.assertEqual(result['maia_elo'], 2600)
        self.assertAlmostEqual(result['players']['white'][0]['expected_accuracy'], .9*100 + .1*first_bad)

    def test_invalid_anchors_are_rejected(self):
        analysis = self.analysis()
        for rating in (True, False, 1600., '1600', None, 500, 1650, 2700):
            with self.subTest(rating=rating), self.assertRaises(ValueError):
                expected_accuracy_by_move(analysis, maia_elo=rating)

    def test_forced_move_is_omitted_without_renumbering_later_points(self):
        result = expected_accuracy_by_move(self.analysis(FORCED_GAP_PGN))
        self.assertEqual([point['move_number'] for point in result['players']['white']], [41, 43])
        self.assertEqual([point['ply'] for point in result['players']['white']], [1, 5])
        self.assertEqual([point['move_number'] for point in result['players']['black']], [41, 42])
        self.assertEqual(result['forced_moves_excluded'], {'white': 1, 'black': 0})

    def test_black_start_uses_fullmove_number_from_position(self):
        fen = chess.STARTING_FEN.replace(' w ', ' b ').rsplit(' ', 1)[0] + ' 37'
        result = expected_accuracy_by_move(self.analysis(f'[FEN "{fen}"]\n\n37... e5 38. Nf3 Nc6 *'))
        self.assertEqual([point['move_number'] for point in result['players']['black']], [37, 38])
        self.assertEqual([point['ply'] for point in result['players']['black']], [1, 3])
        self.assertEqual([point['move_number'] for point in result['players']['white']], [38])
        self.assertEqual([point['ply'] for point in result['players']['white']], [2])

    def test_empty_and_forced_only_games_do_not_fabricate_points(self):
        empty = {'game': game_metadata({}), 'start_fen': chess.STARTING_FEN,
                 'moves': [], 'positions': [{'fen': chess.STARTING_FEN}]}
        forced = self.analysis('[FEN "8/8/8/8/r7/2k5/K7/8 w - - 0 42"]\n\n42. Kb1 *')
        for analysis, excluded in ((empty, 0), (forced, 1)):
            with self.subTest(excluded=excluded), tempfile.TemporaryDirectory() as directory:
                result = expected_accuracy_by_move(analysis)
                self.assertEqual(result['players'], {'white': [], 'black': []})
                self.assertEqual(result['forced_moves_excluded'], {'white': excluded, 'black': 0})
                paths = export_move_curve(analysis, directory)
                ElementTree.parse(paths['curve'])
                self.assertEqual(set(paths), {'curve'})
                self.assertEqual([path.name for path in Path(directory).iterdir()], ['accuracy-by-move-1600.svg'])

    def test_pipeline_and_persistence_keep_all_native_expectations_including_forced_moves(self):
        analysis = self.analysis(FORCED_GAP_PGN)
        expected_ratings = set(map(str, RATINGS))
        for row in analysis['moves']:
            self.assertEqual(set(row['maia']), expected_ratings)
            for profile in row['maia'].values():
                self.assertEqual(set(profile), {'moves', 'expected_accuracy', 'absolute_deviation'})
                self.assertTrue(all(type(profile[key]) is float and math.isfinite(profile[key])
                                    and 0 <= profile[key] <= 100
                                    for key in ('expected_accuracy', 'absolute_deviation')))
                self.assertIsInstance(profile['moves'], list)
                self.assertLessEqual(len(profile['moves']), 5)
                self.assertTrue(all(set(choice) == {'move', 'san', 'eval', 'loss', 'p'}
                                    for choice in profile['moves']))
            for candidate in row['candidate_moves']:
                self.assertEqual(set(candidate['maia_p']), expected_ratings)
        forced = analysis['moves'][2]
        self.assertEqual(chess.Board(forced['fen']).legal_moves.count(), 1)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'analysis.json'
            store = AnalysisStore(Path(directory)/'cache')
            store.save(source, {key: value for key, value in analysis.items()
                                if key not in ('positions', 'position_references')})
            saved = json.loads(source.read_text(encoding='utf-8'))
            loaded = AnalysisStore(Path(directory)/'missing-cache').load(source)
        self.assertEqual(saved['moves'], analysis['moves'])
        self.assertEqual(loaded['moves'], analysis['moves'])
        self.assertNotIn('positions', saved)
        self.assertNotIn('positions', loaded)

    def test_missing_or_invalid_saved_expectation_requires_refresh(self):
        analysis = self.analysis()
        for value in (None, True, '99', float('nan'), float('inf'), -1., 101.):
            with self.subTest(value=value):
                invalid = copy.deepcopy(analysis)
                invalid['moves'][0]['maia']['1600']['expected_accuracy'] = value
                with self.assertRaisesRegex(ValueError, '(?i)refresh'):
                    expected_accuracy_by_move(invalid)
        for missing_profile in (False, True):
            with self.subTest(missing_profile=missing_profile):
                invalid = copy.deepcopy(analysis)
                if missing_profile:
                    del invalid['moves'][0]['maia']['1600']
                else:
                    del invalid['moves'][0]['maia']['1600']['expected_accuracy']
                with self.assertRaisesRegex(ValueError, '(?i)refresh'):
                    expected_accuracy_by_move(invalid)
        for container in (None, [], 'invalid'):
            with self.subTest(container=container):
                invalid = copy.deepcopy(analysis)
                invalid['moves'][0]['maia']['1600'] = container
                with self.assertRaisesRegex(ValueError, '(?i)refresh'):
                    expected_accuracy_by_move(invalid)

    def test_graph_reads_saved_values_without_positions_or_recalculation(self):
        analysis = self.weighted_analysis()
        analysis['moves'][0]['maia']['1600']['expected_accuracy'] = 35.5
        analysis['moves'][0]['accuracy'] = 17.25
        del analysis['positions']
        before = copy.deepcopy(analysis)
        with patch('analysis.accuracy.service.collect_evidence',
                   side_effect=AssertionError('Graph must read persisted expectations')):
            result = expected_accuracy_by_move(analysis)
            self.assertEqual(result['players']['white'][0]['expected_accuracy'], 35.5)
            self.assertEqual(result['players']['white'][0]['accuracy'], 17.25)
            for side, points in result['players'].items():
                for point in points:
                    self.assertEqual(point['accuracy'], analysis['moves'][point['ply']-1]['accuracy'])
            with tempfile.TemporaryDirectory() as directory:
                paths = export_move_curve(analysis, directory)
                ElementTree.parse(paths['curve'])
                self.assertEqual(set(paths), {'curve'})
                self.assertEqual([path.name for path in Path(directory).iterdir()], ['accuracy-by-move-1600.svg'])
        self.assertEqual(analysis, before)

    def test_missing_or_invalid_played_accuracy_is_not_fabricated(self):
        analysis = self.analysis()
        for value in (None, True, '99', float('nan'), float('inf'), -1., 101.):
            with self.subTest(value=value):
                invalid = copy.deepcopy(analysis)
                invalid['moves'][0]['accuracy'] = value
                with self.assertRaises(ValueError):
                    expected_accuracy_by_move(invalid)
        del analysis['moves'][0]['accuracy']
        with self.assertRaises(ValueError):
            expected_accuracy_by_move(analysis)

    def test_position_means_match_the_existing_side_curve(self):
        for analysis in (self.weighted_analysis(), self.analysis(FORCED_GAP_PGN)):
            refresh_saved_curve(analysis)
            for rating in (600, 1600, 2600):
                result = expected_accuracy_by_move(analysis, maia_elo=rating)
                for side in ('white', 'black'):
                    with self.subTest(rating=rating, side=side):
                        points = result['players'][side]
                        expected = analysis['accuracy_curve']['players'][side]['expected_accuracy'][RATINGS.index(rating)]
                        self.assertAlmostEqual(mean(point['expected_accuracy'] for point in points), expected)

    def test_four_figures_and_shared_curve_rebuild_from_public_json_without_cache(self):
        for original in (self.weighted_analysis(), self.analysis(FORCED_GAP_PGN),
                         self.analysis('1. e4 *'), self.analysis('*'),
                         self.analysis('[FEN "8/8/8/8/r7/2k5/K7/8 w - - 0 42"]\n\n42. Kb1 *')):
            with self.subTest(moves=len(original['moves'])), tempfile.TemporaryDirectory() as directory:
                source = Path(directory)/'analysis.json'
                # This fixture deliberately exports prepared-only data into an
                # independent cache; raw pins belong to its source session.
                AnalysisStore(Path(directory)/'cache').save(source, {
                    key: value for key, value in original.items()
                    if key not in ('positions', 'position_references')})
                public = json.loads(source.read_text(encoding='utf-8'))
                self.assertNotIn('positions', public)
                unavailable = AnalysisStore(Path(directory)/'missing-cache')
                prepared = unavailable.load(source)
                self.assertNotIn('positions', prepared)
                self.assertIsNone(unavailable.load_positions(source))
                del prepared['accuracy_curve']
                with patch('analysis.accuracy.service.collect_evidence',
                           side_effect=AssertionError('Prepared data must not rescore raw positions')):
                    refresh_saved_curve(prepared)
                    self.assertEqual(prepared['accuracy_curve'], original['accuracy_curve'])
                    paths = export_saved_figures(prepared, Path(directory)/'figures')
                self.assertEqual(set(paths), {'curve', 'by_move', 'by_move_1800', 'by_move_2000'})
                for path in paths.values():
                    root = ElementTree.parse(path).getroot()
                    self.assertTrue(root.findall('.//{http://www.w3.org/2000/svg}text'),
                                    'Every default SVG must keep selectable text.')
                self.assertEqual({path.name for path in (Path(directory)/'figures').iterdir()},
                                 set(DEFAULT_FIGURE_NAMES))
                for index, rating in enumerate(RATINGS):
                    plotted = expected_accuracy_by_move(prepared, maia_elo=rating)
                    side_means, side_deviations = [], []
                    for side in ('white', 'black'):
                        rows = [row for row in prepared['moves'] if row['side'] == side
                                and chess.Board(row['fen']).legal_moves.count() > 1]
                        summary = prepared['accuracy_curve']['players'][side]
                        self.assertEqual(summary['moves_used'], len(rows))
                        if not rows:
                            self.assertIsNone(summary['expected_accuracy'][index])
                            self.assertIsNone(summary['absolute_deviation'][index])
                            self.assertIsNone(summary['average_accuracy'])
                            continue
                        expected = mean(point['expected_accuracy'] for point in plotted['players'][side])
                        deviation = mean(row['maia'][str(rating)]['absolute_deviation'] for row in rows)
                        self.assertAlmostEqual(summary['expected_accuracy'][index], expected)
                        self.assertAlmostEqual(summary['absolute_deviation'][index], deviation)
                        self.assertAlmostEqual(summary['average_accuracy'], mean(row['accuracy'] for row in rows))
                        side_means.append(expected)
                        side_deviations.append(deviation)
                    if side_means:
                        self.assertAlmostEqual(prepared['accuracy_curve']['expected_accuracy'][index], mean(side_means))
                        self.assertAlmostEqual(prepared['accuracy_curve']['absolute_deviation'][index], mean(side_deviations))
                    else:
                        self.assertIsNone(prepared['accuracy_curve']['expected_accuracy'][index])
                        self.assertIsNone(prepared['accuracy_curve']['absolute_deviation'][index])

    def test_export_separates_three_comparisons_and_replaces_only_its_svg(self):
        from matplotlib.figure import Figure

        analysis = self.analysis(FORCED_GAP_PGN)
        for index, move in enumerate(analysis['moves']):
            move['accuracy'] = 100. if index == 0 else 10.+index
            move['maia']['1600']['expected_accuracy'] = 0. if index == 0 else 70.+index
        expected = expected_accuracy_by_move(analysis)
        plotted = []
        original_savefig = Figure.savefig

        def capture(figure, *args, **kwargs):
            plotted.append(list(figure.axes))
            return original_savefig(figure, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target/'accuracy-by-move-1600.svg').write_text('stale', encoding='utf-8')
            (target/'user-diagram.svg').write_text('keep', encoding='utf-8')
            with patch.object(Figure, 'savefig', autospec=True, side_effect=capture):
                paths = export_move_curve(analysis, directory)
            self.assertEqual(Path(paths['curve']), target/'accuracy-by-move-1600.svg')
            self.assertEqual(set(paths), {'curve'})
            self.assertEqual({path.name for path in target.iterdir()}, {'accuracy-by-move-1600.svg', 'user-diagram.svg'})
            root = ElementTree.parse(paths['curve']).getroot()
            self.assertEqual(root.tag, '{http://www.w3.org/2000/svg}svg')
            text_nodes = root.findall('.//{http://www.w3.org/2000/svg}text')
            self.assertTrue(text_nodes, 'SVG labels must remain selectable text, not glyph paths.')
            self.assertIn('White actual', ''.join(''.join(node.itertext()) for node in text_nodes))
            self.assertEqual((target/'user-diagram.svg').read_text(encoding='utf-8'), 'keep')
        self.assertEqual(len(plotted), 1)
        axes = plotted[0]
        self.assertEqual(len(axes), 3)
        comparisons = ((('white', 'expected_accuracy'), ('black', 'expected_accuracy')),
                       (('white', 'expected_accuracy'), ('white', 'accuracy')),
                       (('black', 'expected_accuracy'), ('black', 'accuracy')))
        for axis, comparison in zip(axes, comparisons, strict=True):
            lines = {line.get_label(): line for line in axis.lines if not line.get_label().startswith('_')}
            wanted_labels = []
            for side, metric in comparison:
                points = expected['players'][side]
                label = (f'{side.title()} Maia 1600: {len(points)} pos'
                         if metric == 'expected_accuracy' else f'{side.title()} actual')
                wanted_labels.append(label)
                line = lines[label]
                self.assertEqual(line.get_color(), '#b25b36' if side == 'white' else '#167d95')
                self.assertEqual(line.get_linestyle(), '-')
                self.assertEqual(line.get_marker(), 'o' if metric == 'expected_accuracy' else 's')
                samples = [(x, y) for x, y in zip(line.get_xdata(), line.get_ydata())
                           if math.isfinite(x) and math.isfinite(y)]
                self.assertEqual(samples, [(p['move_number'], p[metric]) for p in points])
                # A missing forced point must break the solid path, not shift its x coordinate.
                vertices = list(zip(line.get_xdata(), line.get_ydata()))
                for left, right in zip(vertices, vertices[1:]):
                    if all(math.isfinite(value) for value in (*left, *right)):
                        self.assertLessEqual(right[0]-left[0], 1)
            self.assertCountEqual(lines, wanted_labels)
            bridges = [line for line in axis.lines if line.get_linestyle() == ':']
            white_metrics = [metric for side, metric in comparison if side == 'white']
            self.assertEqual(len(bridges), len(white_metrics))
            for bridge in bridges:
                self.assertEqual(list(bridge.get_xdata()), [41, 43])
            self.assertCountEqual([tuple(line.get_ydata()) for line in bridges],
                                  [tuple(p[metric] for p in expected['players']['white'])
                                   for metric in white_metrics])
            self.assertLess(axis.get_ylim()[0], 0, 'A 0% point needs room for its stroke and marker.')
            self.assertGreater(axis.get_ylim()[1], 100, 'A 100% point must not be clipped at the top.')
            self.assertEqual(list(axis.get_yticks()), list(range(0, 101, 20)))
            self.assertTrue(axes[0].get_shared_x_axes().joined(axes[0], axis))
            self.assertTrue(axes[0].get_shared_y_axes().joined(axes[0], axis))
        for side, axis in zip(('white', 'black'), axes[1:], strict=True):
            lines = {line.get_label(): line for line in axis.lines}
            maia = lines[f'{side.title()} Maia 1600: {len(expected["players"][side])} pos']
            actual = lines[f'{side.title()} actual']
            self.assertGreaterEqual(actual.get_alpha(), .65, 'Played accuracy needs readable contrast.')
            self.assertLess(actual.get_alpha(), maia.get_alpha() or 1.)
        self.assertGreater(axes[0].get_position().y0, axes[1].get_position().y1)
        self.assertGreater(axes[1].get_position().y0, axes[2].get_position().y1)

    def test_plot_black_start_keeps_both_sides_on_pgn_move_coordinates(self):
        from matplotlib.figure import Figure

        fen = chess.STARTING_FEN.replace(' w ', ' b ').rsplit(' ', 1)[0] + ' 37'
        analysis = self.analysis(f'[FEN "{fen}"]\n\n37... e5 38. Nf3 Nc6 *')
        plotted = []
        original_savefig = Figure.savefig

        def capture(figure, *args, **kwargs):
            plotted.extend(list(axis.lines) for axis in figure.axes)
            return original_savefig(figure, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Figure, 'savefig', autospec=True, side_effect=capture):
                export_move_curve(analysis, directory)
        self.assertEqual(len(plotted), 3)
        expected_labels = ({'Black Maia 1600: 2 pos': [37, 38], 'White Maia 1600: 1 pos': [38]},
                           {'White Maia 1600: 1 pos': [38], 'White actual': [38]},
                           {'Black Maia 1600: 2 pos': [37, 38], 'Black actual': [37, 38]})
        for lines, expected in zip(plotted, expected_labels, strict=True):
            self.assertEqual({line.get_label(): list(line.get_xdata()) for line in lines}, expected)

    def test_cli_reads_saved_json_and_uses_default_output_directory(self):
        analysis = self.weighted_analysis()
        del analysis['positions']
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'analysis.json'
            source.write_text(json.dumps(analysis), encoding='utf-8')
            original = source.read_bytes()
            with patch('sys.stdout', new_callable=io.StringIO):
                main([str(source), '--maia-elo', '2600'])
            target = source.parent
            ElementTree.parse(target/'accuracy-by-move-2600.svg')
            self.assertEqual({path.name for path in target.iterdir()}, {'analysis.json', 'accuracy-by-move-2600.svg'})
            self.assertEqual(source.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
