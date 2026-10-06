"""Diagram positions, branch provenance and rendering without extra searches."""
import io
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import chess
import chess.pgn
import chess.svg

from coach.tools_chess import ChessTools
from analysis.game.pipeline import analyze_game
from analysis.game.history import history_at
from tests.coach.fixtures import FakeEngines


class DiagramTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.engines = FakeEngines()
        self.addCleanup(self.engines._temp.cleanup)

    def library(self, pgn='1. e4 e5 2. Nf3 Nc6 *', side='white'):
        game = chess.pgn.read_game(io.StringIO(pgn))
        analysis = analyze_game(game, self.engines, side, 1400, progress=lambda _: None)
        return ChessTools(analysis, self.engines, self.directory)

    def assert_svg(self, library, path):
        self.assertIn(path, library.diagrams.paths)
        root = ET.parse(self.directory/path).getroot()
        self.assertEqual(root.tag, '{http://www.w3.org/2000/svg}svg')

    def test_comparison_returns_correct_start_and_both_result_boards(self):
        library = self.library()
        before = len(self.engines.sf_calls)
        with patch('coach.report_diagrams.chess.svg.board', wraps=chess.svg.board) as render:
            result = library.call('compare_played_vs_candidate', {'ply': 3, 'candidate': 'b1c3'})
        # Two candidate checks plus one novel human-reply batch. The played
        # branch reuses saved reply scores; drawing adds no engine searches.
        self.assertEqual(len(self.engines.sf_calls)-before, 3)
        self.assertEqual(render.call_count, 3)
        paths = [result['comparison_diagram']]
        for index, name in enumerate(('played', 'candidate')):
            branch = result[name]
            paths.append(branch['after_defense_diagram'])
            board = self.engines.board(history_at(library.analysis, **branch['after_defense']))
            self.assertEqual(render.call_args_list[index].args[0].fen(), board.fen())
            self.assertEqual(branch['fen_after_defense'], board.fen())
            self.assertEqual(render.call_args_list[index].kwargs['lastmove'], board.peek())
        self.assertEqual(len(set(paths)), 3)
        for path in paths:
            self.assert_svg(library, path)
        root = render.call_args_list[-1]
        self.assertEqual(root.args[0].fen(), library.analysis['moves'][2]['fen'])
        self.assertEqual([a.color for a in root.kwargs['arrows']], ['#189b55cc', '#e69138cc'])
        library.diagrams.validate('\n'.join(f'![Checked position]({p})' for p in paths))

    def test_get_position_renders_later_line_or_reuses_result_without_engine_calls(self):
        library = self.library()
        branch = library.explore_candidate(3, 'b1c3')
        with patch.object(self.engines, 'sf', side_effect=AssertionError('Rendering must not search')), \
             patch.object(self.engines, 'human', side_effect=AssertionError('Rendering must not run Maia')):
            same = library.get_position(**branch['after_defense'])
            self.assertEqual(same['diagram'], branch['after_defense_diagram'])
            line = branch['stockfish']['line']
            endpoint = library.get_position(3, line)
            self.assertEqual(endpoint['fen'], self.engines.board(['e2e4', 'e7e5']+line).fen())
            self.assert_svg(library, endpoint['diagram'])
            earlier = library.get_position(1)
            self.assert_svg(library, earlier['diagram'])

    def test_mating_branch_draws_final_board_with_check_and_correct_orientation(self):
        library = self.library('1. f3 e5 2. g4 Qh4# 0-1', side='black')
        with patch('coach.report_diagrams.chess.svg.board', wraps=chess.svg.board) as render:
            result = library.explore_candidate(4, 'd8h4')
        self.assertTrue(result['checkmate'])
        self.assertIsNone(result['best_defense'])
        self.assertIsNone(result['maia_after_defense'])
        self.assertEqual(result['after_defense']['line'], ['d8h4'])
        board = render.call_args.args[0]
        self.assertTrue(board.is_checkmate())
        self.assertEqual(render.call_args.kwargs['check'], chess.E1)
        self.assertFalse(render.call_args.kwargs['orientation'])
        self.assert_svg(library, result['after_defense_diagram'])

    def test_played_arrow_is_orange_and_orientation_has_a_separate_path(self):
        library = self.library()
        with patch('coach.report_diagrams.chess.svg.board', wraps=chess.svg.board) as render:
            white = library.get_position(1)
            self.assertEqual(render.call_args.kwargs['arrows'][0].color, '#e69138cc')
            library.analysis['selected_player']['side'] = 'black'
            black = library.get_position(1)
            self.assertFalse(render.call_args.kwargs['orientation'])
        self.assertNotEqual(white['diagram'], black['diagram'])


if __name__ == '__main__':
    unittest.main()
