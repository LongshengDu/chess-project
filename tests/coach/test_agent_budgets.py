"""Offline regressions: compact evidence, bounded repairs and the Codex SDK bridge."""
import json
import tempfile
import unittest

from coach.report_output import validate_report

from coach.tools_evidence import prepare_initial_evidence
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from coach.agent_runner import run_coach
from coach.tools_chess import ChessTools
from analysis.game.pipeline import analyze_game
from analysis.game.history import history_at
from analysis.game.study import load_game
from coach.agent_codex import CodexChessSession, run_codex
from coach.agent_budget import CoachingLimitError, RunBudget
from coach.report_output import ReportGate, check_sections
from coach.agent_progress import CoachProgress
from tests.coach.smoke_scenario import run_smoke, validate_smoke_report
from tests.coach.fixtures import FakeEngines, ScriptedCodex


class BudgetTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        pgn = self.root/'game.pgn'
        pgn.write_text('[WhiteElo "1400"]\n[BlackElo "1500"]\n\n1. e4 e5 2. Nf3 Nc6 *', encoding='utf-8')
        self.engines = FakeEngines()
        self.addCleanup(self.engines._temp.cleanup)
        self.analysis = analyze_game(load_game(pgn), self.engines, 'white', 1400, progress=lambda _: None)

    def library(self, smoke=False):
        return ChessTools(self.analysis, self.engines, self.root/'run', max_calls=3 if smoke else 20)

    def test_compact_evidence_retains_scores_and_full_branch_history(self):
        library = self.library()
        compact = library.call('compare_played_vs_candidate', {'ply': 3, 'candidate': 'b1c3'})
        full = library.results[compact['result_id']]
        self.assertLess(len(json.dumps(compact)), len(json.dumps(full))*.65)
        self.assertEqual(compact['candidate']['after_defense'], full['candidate']['after_defense'])
        self.assertEqual(history_at(self.analysis, **compact['candidate']['after_defense'])[:3], ['e2e4','e7e5','b1c3'])
        for score in compact['baseline']['candidate_moves']:
            original = next(c for c in full['baseline']['candidate_moves'] if c['move'] == score['move'])
            self.assertEqual(score['eval'], original['eval'])
            self.assertEqual(score['loss'], original['loss'])
        self.assertEqual(len(compact['baseline']['maia']), 4)
        self.assertEqual(len(full['baseline']['maia']), 17)

    def test_heading_aliases_and_all_missing_sections_are_reported_together(self):
        check_sections('### **1. Ratings**\n## Strengths\n## Weak decisions\n## Key lessons\n## Next steps\n## Practice positions')
        with self.assertRaisesRegex(ValueError, 'performance, main pattern, improvement plan, exercises'):
            check_sections('Body text mentioning ## Best decisions without a real heading.')

    def test_best_and_worst_decisions_are_conditional_not_forced_praise(self):
        required = '## Performance snapshot\n## Main pattern\n## Improvement plan\n## Exercises'
        for optional in ('', '\n## Best decisions', '\n## Worst decisions',
                         '\n## Best decisions\n## Worst decisions'):
            check_sections(required + optional)
        library = self.library()
        prepare_initial_evidence(library)
        image = next(iter(library.diagrams.paths))
        report = required + f'\n![Position]({image})\n' + (
            'The fixture cannot establish a positional improvement or a higher-rated plan. '
            'Use the checked branches to inspect the consequences of each decision. ')*3
        # Both omissions are valid when unsupported; no extra repair/model call.
        self.assertTrue(validate_report(library, report))

    def test_repeated_baseline_is_referenced_but_full_evidence_stays_local(self):
        library = self.library()
        first = library.call('get_position', {'ply': 3})
        later = library.call('compare_played_vs_candidate', {'ply': 3, 'candidate': 'b1c3'})
        self.assertEqual(first['baseline']['ply'], 3)
        self.assertEqual(later['baseline_ref'], 3)
        self.assertNotIn('baseline', later)
        self.assertEqual(library.results[later['result_id']]['baseline']['ply'], 3)
        self.assertGreater(library.invocations[-1]['response_chars'], 0)

    def test_smoke_does_not_require_other_stages_of_a_long_game(self):
        library = self.library(True)
        self.analysis['moves'][0]['stage'] = 'middlegame'
        library.call('compare_played_vs_candidate', {'ply':3,'candidate':'b1c3'})
        image = next(iter(library.diagrams.paths))
        answer = f'# One decision\n![Position]({image})\n' + 'Inspect this verified decision and compare both resulting positions. '*4
        self.assertTrue(validate_smoke_report(library, answer))
        with self.assertRaisesRegex(ValueError,'covering'):
            validate_report(library, answer)




    def test_codex_dynamic_tools_and_short_report_without_any_api(self):
        library = self.library(smoke=True)
        progress_messages = []
        library.progress = CoachProgress(progress_messages.append)
        budget = RunBudget()
        gate = ReportGate(library, report_name='coaching-smoke', max_attempts=1, validator=lambda answer: validate_smoke_report(library, answer))
        clients = []
        class Client:
            def __init__(self, **kwargs):
                self.handler = kwargs['approval_handler']; self.closed = False; self.requests = []
                clients.append(self)
            def start(self): pass
            def initialize(self): pass
            def close(self): self.closed = True
            def request(self, method, params, **kwargs):
                self.requests.append((method, params))
                if method == 'account/read':
                    return NS(model_dump=lambda **k: {'account': {'type':'chatgpt'}})
                return NS(thread=NS(id='fake-thread'), model='fixture')
        class Thread:
            def __init__(self, client, ident): self.client = client
            def turn(self, prompt, **kwargs): return self
            def stream(self):
                public = NS(method='item/completed', payload=NS(item=NS(type='agentMessage',
                    id='public-comment', phase='commentary', text='I am checking the earlier setup before comparing the defenses.')))
                yield public
                yield public  # Replay must not duplicate the public update.
                yield NS(method='item/reasoning/textDelta', payload=NS(delta='PRIVATE_REASONING_SENTINEL'))
                yield NS(method='item/completed', payload=NS(item=NS(type='reasoning',
                    summary='PRIVATE_SUMMARY', content='PRIVATE_REASONING_SENTINEL')))
                for name, args in [('get_position', {'ply':3}), ('compare_played_vs_candidate', {'ply':3,'candidate':'b1c3'})]:
                    response = self.client.handler('item/tool/call', {'tool':name,'arguments':args})
                    assert response['success']
                image = next(iter(library.diagrams.paths))
                answer = f'# Short review\n\n![Position]({image})\n\nActual Elo 1400. This small test compared the played move and a legal alternative through their best defenses. The fixture cannot establish a strategic advantage. Exercise: replay both branches and identify the changed piece placement.'
                yield NS(method='thread/tokenUsage/updated', payload=NS(token_usage=NS(total=NS(input_tokens=100, output_tokens=60, cached_input_tokens=20))))
                yield NS(method='item/completed', payload=NS(item=NS(type='agentMessage', phase='final_answer', text=answer)))
                yield NS(method='turn/completed', payload=NS(turn=NS(status='completed')))
        with patch('openai_codex.Thread', Thread):
            report = run_codex(library, gate, budget, 'fixture', 'fixture', client_factory=Client)
        gate.publish(report)
        self.assertTrue(clients[0].closed)
        self.assertTrue((library.directory/'coaching-smoke.md').exists())
        self.assertFalse((library.directory/'coaching.md').exists())
        self.assertEqual(budget.input_tokens, 100)
        self.assertEqual(sum('earlier setup' in message for message in progress_messages), 1)
        self.assertNotIn('PRIVATE', str(progress_messages))
        self.assertNotIn('# Short review', str(progress_messages))
        self.assertNotIn('earlier setup', report)
        params = clients[0].requests[-1][1]
        self.assertEqual(len(params['dynamicTools']), 9)
        self.assertTrue(params['ephemeral'])
        self.assertFalse(params['config']['features.shell_tool'])

    def test_codex_rejects_api_key_auth_without_starting_a_model_turn(self):
        class Client:
            def __init__(self, **kwargs): self.closed=False; self.methods=[]
            def start(self): pass
            def initialize(self): pass
            def close(self): self.closed=True
            def request(self, method, params, **kwargs):
                self.methods.append(method)
                return NS(model_dump=lambda **k: {'account':{'type':'apiKey'}})
        client = Client()
        library = self.library(True)
        with self.assertRaisesRegex(ValueError, 'ChatGPT sign-in'):
            run_codex(library, ReportGate(library, report_name='coaching-smoke', max_attempts=1, validator=lambda answer: validate_smoke_report(library, answer)), RunBudget(), '', '', allow_tools=False, client_factory=lambda **k:client)
        self.assertEqual(client.methods, ['account/read'])
        self.assertTrue(client.closed)

    def test_codex_repair_respects_response_limit_and_counts_each_unreported_turn(self):
        for limit, report_first_usage in ((1, False), (2, False), (2, True)):
            with self.subTest(limit=limit, report_first_usage=report_first_usage):
                library = self.library(True)
                budget = RunBudget(max_model_responses=limit, max_tokens=100000)

                def reject(answer):
                    raise ValueError('The fixture draft needs repair.')

                gate = ReportGate(library, max_attempts=2, validator=reject)

                class Client:
                    def __init__(self, **kwargs): self.closed = False; self.turns = 0
                    def start(self): pass
                    def initialize(self): pass
                    def close(self): self.closed = True
                    def request(self, method, params, **kwargs):
                        if method == 'account/read':
                            return NS(model_dump=lambda **k: {'account': {'type': 'chatgpt'}})
                        return NS(thread=NS(id='fixture'), model='fixture')

                class Thread:
                    def __init__(self, client, ident): self.client = client
                    def turn(self, prompt, **kwargs): self.client.turns += 1; return self
                    def stream(self):
                        if report_first_usage and self.client.turns == 1:
                            yield NS(method='thread/tokenUsage/updated', payload=NS(token_usage=NS(
                                total=NS(input_tokens=100, output_tokens=50, cached_input_tokens=0))))
                        yield NS(method='item/completed', payload=NS(item=NS(
                            type='agentMessage', phase='final_answer', text='Fixture draft.')))
                        yield NS(method='turn/completed', payload=NS(turn=NS(status='completed')))

                client = Client()
                with patch('openai_codex.Thread', Thread), self.assertRaises(CoachingLimitError):
                    run_codex(library, gate, budget, 'fixture', 'fixture', allow_tools=False,
                              client_factory=lambda **kwargs: client)
                self.assertEqual(client.turns, limit)
                self.assertEqual(budget.requests, limit)
                self.assertEqual(budget.unknown_usage_responses, limit-int(report_first_usage))
                self.assertGreater(budget.estimated_tokens, 0)
                self.assertTrue(client.closed)

    def test_tool_limit_and_unknown_calls_do_not_execute(self):
        session = CodexChessSession(self.library(True), RunBudget())
        for _ in range(3):
            self.assertFalse(session.handle_request('item/tool/call', {'tool':'not-a-tool','arguments':{}})['success'])
        self.assertFalse(session.handle_request('item/tool/call', {'tool':'get_position','arguments':{'ply':3}})['success'])
        self.assertIn('limit', session.stop_reason)
        self.assertEqual(session.library.invocations, [])

    def test_disabled_model_tools_cannot_execute_local_calls(self):
        library = self.library()
        session = CodexChessSession(library, RunBudget(), allow_tools=False)
        response = session.handle_request('item/tool/call', {'tool': 'get_position', 'arguments': {'ply': 3}})
        self.assertFalse(response['success'])
        self.assertEqual(library.invocations, [])
        self.assertEqual(session.tools, {})


    def test_report_rejections_stop_after_one_repair_and_keep_draft(self):
        class BadFinal(ScriptedCodex):
            def report(self, *args, **kwargs):
                return super().report(*args, **kwargs).replace('## Exercises', '## Other')
        model = BadFinal()
        with self.assertRaisesRegex(CoachingLimitError, 'after 2 drafts'):
            model.run(self.analysis, self.engines, self.root/'failed')
        self.assertEqual(model.index, 7)
        self.assertTrue((self.root/'failed/coaching.draft.md').exists())
        self.assertFalse((self.root/'failed/coaching.md').exists())

    def test_unknown_usage_counts_toward_cumulative_budget(self):
        budget = RunBudget(max_model_responses=2, max_tokens=300)
        budget.record_usage(estimate=200)
        budget.check()
        budget.record_usage(estimate=200)
        self.assertEqual(budget.unknown_usage_responses, 2)
        with self.assertRaisesRegex(CoachingLimitError, 'token budget'):
            budget.check()

    def test_smoke_prepares_one_decision_and_requests_only_one_short_answer(self):
        output = self.root/'brief'
        output.mkdir()
        (output/'coaching.md').write_text('previous accepted report', encoding='utf-8')
        model = ScriptedCodex()
        with patch('coach.agent_codex.run_codex', side_effect=model):
            report = run_smoke(self.analysis, self.engines, output)
        self.assertEqual(model.index, 1)
        self.assertLess(len(report.split()), 250)
        self.assertEqual((output/'coaching.md').read_text(encoding='utf-8'), 'previous accepted report')
        self.assertTrue((output/'coaching-smoke.md').exists())
        self.assertEqual(len(self.analysis['agent_run']['investigated_plies']), 1)
        self.assertEqual(self.analysis['agent_run']['tool_calls'], 2)
        self.assertTrue((output/'usage.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
