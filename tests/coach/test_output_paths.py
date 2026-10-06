"""CLI output follows the input game, regardless of the working directory."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from analysis.game.pipeline import ANALYSIS_VERSION
from analysis.game.study import load_game
from coach.coach import main
from analysis.cache import identity
from tests.coach import benchmark_game_speed


class OutputPathTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pgn = self.root/'session games'/'rapid.study.pgn'
        self.pgn.parent.mkdir()
        self.pgn.write_text('1. e4 e5 *', encoding='utf-8')
        self.output = self.pgn.parent/'output'/'rapid.study-full'
        self.engines = Mock()
        self.engines.__enter__ = Mock(return_value=self.engines)
        self.engines.__exit__ = Mock(return_value=False)

    def run_analysis(self, pgn, *extra):
        with patch('coach.coach.Engines',return_value=self.engines), \
             patch('coach.coach.analyze_game',return_value={'fixture':True}) as analyze, \
             patch('coach.agent_runner.run_coach') as coach:
            self.assertEqual(main([str(pgn),'--side','white','--elo','1600','--analysis-only',*map(str,extra)]),0)
        analyze.assert_called_once()
        expected = Path(extra[1]) if extra and extra[0] == '--output-dir' else Path(pgn).parent/'output'/f'{Path(pgn).stem}-full'
        self.assertEqual(analyze.call_args.kwargs['rating_output_dir'], expected/'player-rating')
        coach.assert_not_called()
        self.engines.__exit__.assert_called_once()

    def test_absolute_game_path_uses_its_parent_and_full_stem(self):
        self.run_analysis(self.pgn)
        self.assertEqual(json.loads((self.output/'analysis.json').read_text(encoding='utf-8')),{'fixture':True})
        self.assertEqual((self.output/'game.pgn').read_text(encoding='utf-8'),self.pgn.read_text(encoding='utf-8'))
        self.assertFalse((self.root/'coach'/'output').exists())

    def test_relative_game_path_is_independent_of_repository_root(self):
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            self.run_analysis(self.pgn.relative_to(self.root))
        finally:
            os.chdir(previous)
        self.assertTrue((self.output/'analysis.json').is_file())

    def test_explicit_output_override_remains_exact(self):
        custom = self.root/'custom output'
        self.run_analysis(self.pgn,'--output-dir',custom)
        self.assertTrue((custom/'analysis.json').is_file())
        self.assertFalse(self.output.exists())

    def test_coach_only_reads_the_same_game_specific_default(self):
        game = load_game(self.pgn)
        self.output.mkdir(parents=True)
        saved = {'schema_version':ANALYSIS_VERSION,
                 'game_id':identity([game.board().fen(),[m.uci() for m in game.mainline_moves()]]),
                 'selected_player':{'side':'white','actual_elo':1600}}
        (self.output/'analysis.json').write_text(json.dumps(saved),encoding='utf-8')
        def coach(analysis,engines,output,**kwargs):
            self.assertEqual(output,self.output)
            analysis['agent_run'] = {'usage':{'total_tokens':0}}
        with patch('coach.coach.Engines',return_value=self.engines), \
             patch('coach.coach.analyze_game') as analyze, \
             patch('analysis.player_rating.service.refresh_saved_rating') as refresh, \
             patch('coach.coach.refresh_performance'), \
             patch('coach.agent_runner.run_coach',side_effect=coach) as run:
            self.assertEqual(main([str(self.pgn),'--side','white','--elo','1600','--coach-only']),0)
        analyze.assert_not_called()
        run.assert_called_once()
        refresh.assert_called_once()
        self.assertEqual(refresh.call_args.kwargs['output_dir'], self.output/'player-rating')
        self.engines.__exit__.assert_called_once()

    def test_profiler_cache_defaults_to_yaml_and_respects_explicit_override(self):
        configured = self.root/'configured cache'
        explicit = self.root/'explicit cache'
        self.engines.signature = {'device':'cpu'}
        self.engines.stats = {}
        self.engines.last_analysis_execution = {'workers':1,'threads_per_worker':2,'hash_mb_per_worker':512}
        for cache,extra in ((configured,[]),(explicit,['--cache-dir',str(explicit)])):
            with self.subTest(cache=cache), \
                 patch.dict(benchmark_game_speed.CONFIG['ANALYSIS'],CACHE_DIR=configured), \
                 patch.object(benchmark_game_speed,'Engines',return_value=self.engines) as constructor, \
                 patch.object(benchmark_game_speed,'analyze_game',return_value={'moves':[]}) as analyze, \
                 patch('builtins.print'):
                benchmark_game_speed.main([str(self.pgn),'--output-dir',str(self.output),*extra])
                self.assertEqual(constructor.call_args.args[1],cache)
                self.assertEqual(analyze.call_args.kwargs['rating_output_dir'], self.output/'player-rating')
                timings = json.loads((self.output/'timing.json').read_text(encoding='utf-8'))
                self.assertEqual(Path(timings['cache_dir']),cache.resolve())
                self.assertFalse((self.output/'cache').exists())


if __name__ == '__main__':
    unittest.main()
