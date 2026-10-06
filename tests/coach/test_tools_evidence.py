"""Offline coverage for preparation, one-response batches, and usage deltas."""
import io
import json
import tempfile
import unittest

from coach.tools_evidence import prepare_initial_evidence
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
from unittest.mock import patch

import chess.pgn
from coach.agent_runner import run_coach
from coach.tools_chess import ChessTools
from analysis.game.pipeline import analyze_game
from analysis.game.summary import compact_summary
from coach.agent_codex import CodexChessSession
from coach.agent_budget import CoachingLimitError, RunBudget
from tests.coach.fixtures import FakeEngines, ScriptedCodex


class PreparedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.engines = FakeEngines()
        self.addCleanup(self.engines._temp.cleanup)
        game = chess.pgn.read_game(io.StringIO('1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *'))
        self.analysis = analyze_game(game, self.engines, 'white', 1400, progress=lambda _: None)

    def library(self, max_calls=20):
        return ChessTools(self.analysis, self.engines, self.root, max_calls=max_calls)

    def test_first_model_response_can_finish_report_without_discovery_calls(self):
        model = ScriptedCodex()
        model.index = 5  # Return a complete report using the supplied diagrams.
        with patch('analysis.move_hints.move_flags', side_effect=AssertionError('Flags must be reused from analysis')):
            report = model.run(self.analysis, self.engines, self.root)
        self.assertEqual(model.index, 6)
        self.assertIn('## Exercises', report)
        self.assertEqual(self.analysis['agent_run']['investigated_plies'], [1, 3, 5])
        initial = json.loads((self.root/'initial_evidence.json').read_text(encoding='utf-8'))
        self.assertEqual(len(initial['initial_evidence']), 3)
        self.assertEqual([w['target_ply'] for w in initial['leadup_context']['windows']],[1,3,5])
        self.assertEqual([m['ply'] for m in initial['leadup_context']['moves']],[1,2,3,4])
        self.assertTrue(all(c['played']['after_defense'] and c['candidate']['after_defense'] for c in initial['initial_evidence']))
        trace = [json.loads(line) for line in (self.root/'agent_trace.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertTrue(all(e['phase']=='initial' for e in trace))
        self.assertEqual(len(trace), 3)

    def test_preparation_covers_quiet_stages_without_recalculating_or_adding_flags(self):
        for row in self.analysis['moves']:
            del row['flags']
            row['stage'] = {1:'opening', 3:'middlegame', 5:'endgame'}.get(row['ply'], 'opening')
        before = json.dumps(self.analysis, sort_keys=True)
        with patch('analysis.move_hints.move_flags', side_effect=AssertionError('Summary must only read saved data')):
            summary = compact_summary(self.analysis)
        self.assertEqual({m['stage'] for m in summary['critical_moments']}, {'opening','middlegame','endgame'})
        self.assertEqual(json.dumps(self.analysis, sort_keys=True), before)
        self.assertTrue(all(m['flags'] == [] for m in summary['critical_moments']))
        library = self.library()
        prepare_initial_evidence(library)
        self.assertEqual(library.investigated,{1,3,5})
        self.assertEqual(json.dumps(self.analysis, sort_keys=True), before)

    def test_batch_preserves_arguments_order_history_and_counts_children(self):
        library = self.library()
        session = CodexChessSession(library, RunBudget())
        response = session.handle_request('item/tool/call', {'tool':'investigate_batch', 'arguments':{'requests':[
            {'tool':'compare_played_vs_candidate','arguments':{'ply':1,'candidate':'d2d4'}},
            {'tool':'get_position','arguments':{'ply':3,'line':['b1c3']}}]}})
        self.assertTrue(response['success'])
        batch = json.loads(response['contentItems'][0]['text'])['results']
        self.assertEqual([r['index'] for r in batch],[0,1])
        self.assertEqual(batch[0]['result']['candidate']['stockfish']['move'],'d2d4')
        self.assertEqual(batch[1]['result']['fen'], self.engines.board(['e2e4','e7e5','b1c3']).fen())
        self.assertEqual(len(library.invocations),2)

    def test_batch_rejects_invalid_schema_before_work_and_budget_cannot_be_bypassed(self):
        library = self.library(1)
        good = {'tool':'get_position','arguments':{'ply':1}}
        for bad in [{'tool':'investigate_batch','arguments':{'requests':[good]}},
                    {'tool':'get_position','arguments':{}}, {'tool':'get_position','arguments':{'ply':'1'}}]:
            with self.assertRaises((ValueError,TypeError)):
                library.call('investigate_batch',{'requests':[good,bad]})
            self.assertEqual(library.invocations,[])
        with self.assertRaises(CoachingLimitError):
            library.call('investigate_batch',{'requests':[good,good]})
        self.assertEqual(library.invocations,[])

    def test_bad_branch_does_not_discard_other_independent_results(self):
        library = self.library()
        result = library.call('investigate_batch',{'requests':[
            {'tool':'get_position','arguments':{'ply':1,'line':['e2e5']}},
            {'tool':'get_position','arguments':{'ply':3}}]})
        self.assertIn('error',result['results'][0])
        self.assertIn('result',result['results'][1])
        self.assertEqual(len(library.invocations),2)

    def test_batch_search_bounds_are_checked_before_any_child_runs(self):
        library = self.library()
        for movetime in (0, self.engines.limits.max_ms + 1):
            with self.subTest(movetime=movetime), self.assertRaises(ValueError):
                library.call('investigate_batch', {'requests': [
                    {'tool': 'get_position', 'arguments': {'ply': 1}},
                    {'tool': 'stockfish_analyze', 'arguments': {'ply': 3, 'movetime_ms': movetime}},
                ]})
            self.assertEqual(library.invocations, [])
            self.assertEqual(library.diagrams.paths, set())

    def test_cached_repeat_comparison_does_not_rerun_engines(self):
        library = self.library()
        initial = json.loads(prepare_initial_evidence(library))
        sample = initial['initial_evidence'][0]
        self.engines.sf = Mock(side_effect=AssertionError('No repeated search'))
        repeat = library.call('compare_played_vs_candidate',
            {'ply':sample['ply'],'candidate':sample['candidate']['stockfish']['move']})
        self.assertEqual(repeat['result_id'],sample['result_id'])
        self.assertEqual(repeat['baseline_ref'],sample['ply'])


class UsageTests(unittest.TestCase):
    def test_codex_cumulative_updates_become_response_deltas_and_ignore_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget = RunBudget(usage_path=Path(tmp)/'response_usage.jsonl')
            context = {'tool_calls_completed':3,'tool_response_chars':5000,'initial_tool_calls':3}
            budget.usage_context = lambda: context
            first = NS(input_tokens=10000,output_tokens=100,cached_input_tokens=0)
            self.assertTrue(budget.record_cumulative(first, first))
            self.assertFalse(budget.record_cumulative(first, first))
            context = {**context,'tool_calls_completed':5,'tool_response_chars':7000}
            second = NS(input_tokens=22000,output_tokens=500,cached_input_tokens=9000)
            budget.record_cumulative(second, NS(input_tokens=12000,output_tokens=400,cached_input_tokens=9000))
            self.assertFalse(budget.record_cumulative(first)) # stale event
            events = [json.loads(line) for line in budget.usage_path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(budget.requests,2)
            self.assertEqual([e['input_tokens'] for e in events],[10000,12000])
            self.assertEqual(events[1]['uncached_input_tokens'],3000)
            self.assertEqual(events[1]['new_tool_calls_completed'],2)
            self.assertEqual(events[1]['new_tool_response_chars'],2000)
            self.assertEqual(sum(e['total_tokens'] for e in events),22500)
            self.assertEqual(budget.input_tokens+budget.output_tokens,22500)
            self.assertEqual(events[1]['provider_last']['input_tokens'],12000)

    def test_known_and_unknown_provider_responses_are_logged_without_content(self):
        budget = RunBudget()
        budget.requests += 1
        budget.record_usage(500,50,200)
        budget.requests += 1
        budget.record_usage(estimate=134)
        self.assertEqual(len(budget.response_usage),2)
        self.assertIsNone(budget.response_usage[1]['total_tokens'])
        self.assertEqual(budget.response_usage[1]['estimated_tokens'],134)
        self.assertEqual(budget.input_tokens,500)


if __name__ == '__main__':
    unittest.main()
