"""Accuracy curve measurements remain descriptive and independent of Elo inputs."""
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

import chess.pgn

from analysis.game.pipeline import analyze_game
from analysis.game.metadata import game_metadata
from analysis.accuracy.evidence import RATINGS, normalized_policies
from analysis.accuracy.figures import (
    DEFAULT_FIGURE_NAMES, _intersection_ranges, _mark_intersections, export_saved_figures,
)
from analysis.accuracy.service import refresh_saved_curve, summarize_evidence
from tests.coach.fixtures import FakeAnalysisSession


def observation(qualities, probabilities, played='a'):
    return {'played': played, 'qualities': qualities,
            'policies': {rating: probabilities.copy() for rating in RATINGS}}


class AccuracyCurveTests(unittest.TestCase):
    def test_long_intersection_legends_fit_text_and_stay_inside_their_panels(self):
        from matplotlib.figure import Figure

        ratings = list(RATINGS)
        expected = [75. if index % 2 == 0 else 85. for index in range(len(ratings))]
        data = {'ratings': ratings, 'expected_accuracy': expected,
                'absolute_deviation': [3.]*len(ratings), 'players': {
                    side: {'average_accuracy': 80., 'expected_accuracy': expected,
                           'lichess_accuracy': 76.25, 'moves_used': 123,
                           'absolute_deviation': [3.]*len(ratings)}
                    for side in ('white', 'black')}}
        captured = []
        original_savefig = Figure.savefig

        def capture(figure, *args, **kwargs):
            if len(figure.axes) == 4:
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                for axis in figure.axes:
                    legend = axis.get_legend()
                    self.assertIsNotNone(legend)
                    frame = legend.get_frame().get_window_extent(renderer)
                    bounds = axis.get_window_extent(renderer)
                    self.assertGreaterEqual(frame.x0, bounds.x0-.5)
                    self.assertGreaterEqual(frame.y0, bounds.y0-.5)
                    self.assertLessEqual(frame.x1, bounds.x1+.5)
                    self.assertLessEqual(frame.y1, bounds.y1+.5)
                    for label in legend.get_texts():
                        text = label.get_window_extent(renderer)
                        self.assertGreaterEqual(text.x0, frame.x0-.5)
                        self.assertGreaterEqual(text.y0, frame.y0-.5)
                        self.assertLessEqual(text.x1, frame.x1+.5)
                        self.assertLessEqual(text.y1, frame.y1+.5)
                    captured.append(' '.join(' '.join(label.get_text().split())
                                             for label in legend.get_texts()))
            return original_savefig(figure, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Figure, 'savefig', autospec=True, side_effect=capture):
                export_saved_figures({'game': game_metadata({}), 'accuracy_curve': data, 'start_fen': chess.STARTING_FEN,
                                      'moves': []}, directory)
        self.assertEqual(len(captured), 4)
        # Wrapping must retain every crossing rather than hiding overflowing evidence.
        for legend in captured[:3]:
            for rating in range(650, 2600, 100):
                self.assertIn(str(rating), legend)
        self.assertIn('White: avg 80.0; lc 76.2; 123 pos', captured[0])
        self.assertIn('Black: avg 80.0; lc 76.2; 123 pos', captured[1])

    def test_four_panels_put_crossings_in_legends_and_overlay_side_curves(self):
        from matplotlib.figure import Figure

        ratings = list(RATINGS)
        data = {'ratings': ratings, 'expected_accuracy': [70+.01*(r-600) for r in ratings],
                'absolute_deviation': [3.]*len(ratings), 'players': {
                    'white': {'average_accuracy': 80., 'expected_accuracy': [76+.01*(r-600) for r in ratings],
                              'lichess_accuracy': 75.25, 'moves_used': 45,
                              'absolute_deviation': [3.]*len(ratings)},
                    'black': {'average_accuracy': 78., 'expected_accuracy': [70+.005*(r-600) for r in ratings],
                              'lichess_accuracy': 70.1, 'moves_used': 44,
                              'absolute_deviation': [3.]*len(ratings)}}}
        captured = []
        original_savefig = Figure.savefig

        def capture(figure, *args, **kwargs):
            if len(figure.axes) == 4:
                captured.extend(figure.axes)
            return original_savefig(figure, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Figure, 'savefig', autospec=True, side_effect=capture):
                export_saved_figures({'game': game_metadata({}), 'accuracy_curve': data, 'start_fen': chess.STARTING_FEN, 'moves': []}, directory)
            svg = (Path(directory)/'accuracy-curve.svg').read_text(encoding='utf-8')
        root = ElementTree.fromstring(svg, parser=ElementTree.XMLParser(
            target=ElementTree.TreeBuilder(insert_comments=True)))
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        text_nodes = root.findall('.//svg:text', ns)
        self.assertTrue(text_nodes, 'SVG labels must remain selectable text, not glyph paths.')
        self.assertIn('White position accuracy curve', ''.join(''.join(node.itertext()) for node in text_nodes))
        self.assertEqual(len(captured), 4)
        for axis in captured:
            self.assertEqual(list(axis.get_xticks()), list(range(600, 2601, 200)))
            lines = {line.get_label(): line for line in axis.lines}
            for side, color in (('White', '#b25b36'), ('Black', '#167d95')):
                if f'{side} positions' in lines:
                    self.assertEqual(lines[f'{side} positions'].get_color(), color)
                    self.assertEqual(lines[f'{side} positions'].get_linestyle(), '-')
            for label in ('Shared curve', 'Shared absolute deviation'):
                if label in lines:
                    self.assertEqual(lines[label].get_linestyle(), ':')
        self.assertEqual(captured[-1].get_ylim(), (0., 50.))

        def label_text(group):
            return '\n'.join(' '.join(''.join(node.itertext()).split()) for node in group.iter()
                             if node.tag == '{http://www.w3.org/2000/svg}text')

        panels = {group.get('id'): label_text(group)
                  for group in root.findall('.//svg:g', ns) if group.get('id', '').startswith('axes_')}
        legends = {group.get('id'): label_text(group)
                   for group in root.findall('.//svg:g', ns) if group.get('id', '').startswith('legend_')}
        self.assertEqual(len(panels), 4)
        expected = (
            ('White position accuracy curve', 'White: avg 80.0; lc 75.2; 45 pos', 'xelo 1000'),
            ('Black position accuracy curve', 'Black: avg 78.0; lc 70.1; 44 pos', 'xelo 2200'),
            ('Shared accuracy curve', 'Shared curve', 'White positions', 'Black positions',
             'xelo 1600', 'xelo 1400',
             'White: avg 80.0; lc 75.2; 45 pos',
             'Black: avg 78.0; lc 70.1; 44 pos'),
            ('Probability-weighted absolute deviation', 'White positions', 'Black positions', 'Shared absolute deviation'),
        )
        for index, labels in enumerate(expected, start=1):
            for label in labels:
                self.assertIn(label, panels[f'axes_{index}'])
            for label in labels[1:]:
                self.assertIn(label, legends[f'legend_{index}'])
        for removed in ('White: 80.00%', 'Black: 78.00%'):
            self.assertNotIn(removed, svg)
        for label in ('Black: avg', 'xelo 1600', 'xelo 2200', 'xelo 1400'):
            self.assertNotIn(label, panels['axes_1'])
        for label in ('White: avg', 'xelo 1000', 'xelo 1600', 'xelo 1400'):
            self.assertNotIn(label, panels['axes_2'])
        for label in ('xelo 1000', 'xelo 2200'):
            self.assertNotIn(label, panels['axes_3'])
        self.assertNotIn('avg ', panels['axes_4'])
        for removed in ('All legal move probabilities',
                        'Intersections are curve coordinates', 'Shading is descriptive spread'):
            self.assertNotIn(removed, svg)

    def test_curve_intersections_follow_drawn_segments_without_extrapolation(self):
        self.assertEqual(_intersection_ranges([600, 700, 800], [70., 80., 90.], 75.), [(650., 650.)])
        self.assertEqual(_intersection_ranges([600, 700, 800], [70., 80., 90.], 80.), [(700., 700.)])
        self.assertEqual(_intersection_ranges([600, 700, 800], [70., 80., 70.], 75.),
                         [(650., 650.), (750., 750.)])
        self.assertEqual(_intersection_ranges([600, 700, 800, 900], [70., 80., 80., 90.], 80.),
                         [(700., 800.)])
        self.assertEqual(_intersection_ranges([600, 700, 800], [70., None, 90.], 80.), [])
        self.assertEqual(_intersection_ranges([600, 700, 800], [70., 80., 90.], 95.), [])

    def test_intersections_use_projection_lines_without_point_or_overlap_markers(self):
        from matplotlib.figure import Figure

        figure = Figure()
        axis = figure.subplots()
        _mark_intersections(axis, [(1000., 1000.), (1600., 1800.)], 80., '#167d95')
        self.assertFalse(axis.lines, 'Crossings need no point marker or heavy overlap segment.')
        segments = [segment.tolist() for collection in axis.collections
                    for segment in collection.get_segments()]
        self.assertEqual(segments, [[[1000., 50.], [1000., 80.]],
                                    [[1600., 50.], [1600., 80.]],
                                    [[1800., 50.], [1800., 80.]]])

    def test_unavailable_unplayed_position_is_not_relabelled_complete(self):
        from analysis.position_results import stockfish_result
        result = stockfish_result(chess.Board(), {'lines': [], 'best_move': None, 'engine_moves': [],
            'search': {'complete': False, 'coverage_complete': False, 'available': False}})
        self.assertFalse(result['complete'])
        self.assertFalse(result['coverage_complete'])
        self.assertFalse(result['available'])

    def test_rare_catastrophe_has_linear_absolute_deviation(self):
        row = observation({'a': 100., 'b': 0.}, {'a': .99, 'b': .01})
        result = summarize_evidence({'white': [row], 'black': []})
        self.assertEqual(result['expected_accuracy'], [99.]*21)
        for value in result['absolute_deviation']:
            self.assertAlmostEqual(value, 1.98)
        self.assertEqual(result['players']['white']['average_accuracy'], 100.)

    def test_deviation_is_by_elo_and_sides_have_equal_weight(self):
        white = observation({'a': 100., 'b': 0.}, {'a': .99, 'b': .01})
        white['policies'][2600] = {'a': .999, 'b': .001}
        black = observation({'a': 80., 'b': 60.}, {'a': .5, 'b': .5})
        result = summarize_evidence({'white': [white]*4, 'black': [black]})
        self.assertAlmostEqual(result['expected_accuracy'][0], (99+70)/2)
        self.assertAlmostEqual(result['absolute_deviation'][0], (1.98+10)/2)
        self.assertAlmostEqual(result['absolute_deviation'][-1], (.1998+10)/2)
        repeated = summarize_evidence({'white': [white]*40, 'black': [black]*10})
        self.assertEqual(result['absolute_deviation'], repeated['absolute_deviation'])

    def test_forced_moves_do_not_inflate_arithmetic_or_curve(self):
        forced = observation({'a': 100.}, {'a': 1.})
        choice = observation({'a': 70., 'b': 50.}, {'a': .5, 'b': .5})
        result = summarize_evidence({'white': [forced, choice], 'black': [forced]})
        self.assertEqual(result['expected_accuracy'], [60.]*21)
        self.assertEqual(result['players']['white']['average_accuracy'], 70.)
        self.assertEqual(result['players']['white']['forced_moves_excluded'], 1)
        self.assertIsNone(result['players']['black']['average_accuracy'])

    def test_full_policy_is_required_even_for_unlikely_moves(self):
        policy = {rating: {'a': 1.} for rating in RATINGS}
        with self.assertRaisesRegex(ValueError, 'complete legal'):
            normalized_policies(policy, {'a', 'b'})

    def test_refresh_is_self_contained_and_exports_replacement_svg(self):
        game = chess.pgn.read_game(io.StringIO('[WhiteElo "1400"]\n[BlackElo "1700"]\n\n1. e4 e5 2. Nf3 Nc6 *'))
        session = FakeAnalysisSession(game.board().fen())
        self.addCleanup(session._temp.cleanup)
        analysis = analyze_game(game, session, progress=lambda _: None)
        original = copy.deepcopy(analysis['accuracy_curve'])
        analysis['headers'].update(WhiteElo='2200', BlackElo='800', WhiteEloEstimate='9999')
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            refresh_saved_curve(analysis, output_dir=directory)
            self.assertEqual(analysis['accuracy_curve'], original)
            svg = directory/'accuracy-curve.svg'
            self.assertIn('Probability-weighted absolute deviation', svg.read_text(encoding='utf-8'))
            svg.write_text('stale', encoding='utf-8')
            refresh_saved_curve(analysis, output_dir=directory)
            self.assertTrue(svg.read_text(encoding='utf-8').startswith('<?xml'))
            for rating in (1600, 1800, 2000):
                move_svg = directory/f'accuracy-by-move-{rating}.svg'
                self.assertIn(f'Maia {rating}', move_svg.read_text(encoding='utf-8'))
            self.assertEqual({p.name for p in directory.iterdir()}, set(DEFAULT_FIGURE_NAMES))
        self.assertEqual(original['ratings'], list(range(600, 2601, 100)))
    def test_pipeline_writes_four_default_figures(self):
        game = chess.pgn.read_game(io.StringIO('1. e4 e5 *'))
        session = FakeAnalysisSession(game.board().fen())
        self.addCleanup(session._temp.cleanup)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            analyze_game(game, session, progress=lambda _: None, accuracy_output_dir=output)
            expected_names = {'accuracy-curve.svg', 'accuracy-by-move-1600.svg',
                              'accuracy-by-move-1800.svg', 'accuracy-by-move-2000.svg'}
            self.assertEqual(set(DEFAULT_FIGURE_NAMES), expected_names)
            self.assertEqual({p.name for p in output.iterdir()}, expected_names)
            for path in output.iterdir():
                ElementTree.parse(path)

    def test_arithmetic_measurement_matches_scored_sequence_including_mate(self):
        from statistics import mean
        for pgn in ('1. f3 e5 2. g4 Qh4# 0-1',
                    '[FEN "4k3/8/8/8/8/8/3q4/4K3 w - - 0 1"]\n\n1. Kf1 Qd1+ 2. Kf2 *'):
            with self.subTest(pgn=pgn):
                game = chess.pgn.read_game(io.StringIO(pgn))
                self.assertFalse(game.errors)
                session = FakeAnalysisSession(game.board().fen())
                self.addCleanup(session._temp.cleanup)
                analysis = analyze_game(game, session, progress=lambda _: None)
                for side in ('white', 'black'):
                    expected = [row['accuracy'] for row in analysis['moves']
                                if row['side'] == side and chess.Board(row['fen']).legal_moves.count() > 1]
                    actual = analysis['accuracy_curve']['players'][side]['average_accuracy']
                    if expected:
                        self.assertAlmostEqual(actual, mean(expected))
                    else:
                        self.assertIsNone(actual)

    def test_missing_complete_scores_is_not_silently_approximated(self):
        game = chess.pgn.read_game(io.StringIO('1. e4 e5 *'))
        session = FakeAnalysisSession(game.board().fen())
        self.addCleanup(session._temp.cleanup)
        analysis = analyze_game(game, session, progress=lambda _: None)
        analysis['positions'][0]['stockfish']['cp_vec'].pop('a2a3')
        with self.assertRaisesRegex(ValueError, 'all legal moves'):
            refresh_saved_curve(analysis)


if __name__ == '__main__':
    unittest.main()
