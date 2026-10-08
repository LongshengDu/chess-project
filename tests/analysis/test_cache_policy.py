"""Requested-limit dominance and history reductions without native engines."""
from copy import deepcopy
import unittest

import chess

from analysis.cache.policy import (dominates, history_compatible, promote_request,
                                   quality_key, request_satisfies)
from analysis.cache.requests import maia_request, stockfish_exploration_request, stockfish_initial_request
from tests.analysis.test_stockfish_search import evaluation_frame


ENGINE = {'stockfish': 'sha256:fixture', 'threads': 4, 'hash_mb': 256}
MAIA = {'maia': 'sha256:fixture', 'history_window': 3, 'device': 'cpu'}


def evaluation(depth=18, seconds=2, candidates=('e2e4',), strategy='bounded'):
    return stockfish_initial_request(ENGINE, depth, seconds, seconds, strategy,
                                    {'forcedCandidateMoves': list(candidates)})


def frame(*, depth=1, strategy='bounded'):
    return evaluation_frame(depth=depth, strategy=strategy)


def board(moves):
    result = chess.Board()
    for move in moves.split():
        result.push_uci(move)
    return result


class RequestPolicyTests(unittest.TestCase):
    def test_bounded_ceiling_does_not_change_request_identity(self):
        first = stockfish_initial_request(ENGINE, 18, 2, 6, 'bounded', {})
        second = stockfish_initial_request(ENGINE, 18, 2, 30, 'bounded', {})
        self.assertEqual(first, second)
        self.assertEqual(first['max_budget_seconds'], 2)

    def test_complete_time_limited_result_reuses_by_requested_limits(self):
        stored, result = evaluation(22, 10), frame(depth=1)
        for requested in (evaluation(22, 10), evaluation(18, 5), evaluation(22, 5), evaluation(18, 10)):
            with self.subTest(requested=requested):
                self.assertTrue(request_satisfies(stored, result, requested))

    def test_either_insufficient_requested_limit_requires_more_work(self):
        stored, result = evaluation(18, 5), frame(depth=40)
        for requested in (evaluation(22, 5), evaluation(18, 10), evaluation(22, 10)):
            self.assertFalse(request_satisfies(stored, result, requested))

    def test_candidate_superset_is_reusable_regardless_order(self):
        stored = evaluation(candidates=('e2e4', 'd2d4', 'c2c4'))
        self.assertTrue(request_satisfies(stored, frame(), evaluation(candidates=('d2d4', 'e2e4'))))
        self.assertFalse(request_satisfies(stored, frame(depth=40), evaluation(candidates=('g1f3',))))

    def test_engine_policy_strategy_and_future_options_remain_distinct(self):
        stored = evaluation()
        variants = [
            {**stored, 'engine': {**ENGINE, key: value}}
            for key, value in (('stockfish', 'sha256:other'), ('threads', 8), ('hash_mb', 512))]
        variants += [{**stored, 'strategy': 'staged'}, {**stored, 'policy_version': 1},
                     {**stored, 'future_option': True}, {**stored, 'engine': {}}]
        for requested in variants:
            self.assertFalse(request_satisfies(stored, frame(depth=40), requested))
        obsolete = {**stored, 'policy_version': 1}
        self.assertFalse(request_satisfies(obsolete, frame(), obsolete))

    def test_complete_coverage_is_required_even_for_exact_requests(self):
        request = evaluation()
        for change in ({'complete': False}, {'coverage_complete': False}, {'root_move_depth_vec': {}},
                       {'cp_vec': {'e2e4': 20}}, {'best_move': None}, {'engine_moves': []}):
            self.assertFalse(request_satisfies(request, {**frame(), **change}, request))

    def test_staged_and_exhaustive_use_effective_requested_watchdog(self):
        for strategy in ('staged', 'exhaustive'):
            stored = stockfish_initial_request(ENGINE, 22, 1, 10, strategy, {})
            requested = stockfish_initial_request(ENGINE, 18, 1, 5, strategy, {})
            self.assertTrue(request_satisfies(stored, frame(strategy=strategy), requested))
            self.assertFalse(request_satisfies(requested, frame(depth=40, strategy=strategy), stored))

    def test_exploration_keeps_multipv_and_root_restrictions(self):
        stored = stockfish_exploration_request(ENGINE, 2000, 22, 1, None)
        result = {'lines': [{'cp': 15, 'mate': None, 'depth': 1, 'pv_uci': ['e2e4']}]}
        requested = stockfish_exploration_request(ENGINE, 1000, 18, 1, None)
        self.assertTrue(request_satisfies(stored, result, requested))
        self.assertFalse(request_satisfies(requested, result, stored))
        for change in ({'multipv': 2}, {'root_moves': ['e2e4']}, {'version': 1}, {'depth': 24}):
            self.assertFalse(request_satisfies(stored, result, {**requested, **change}))
        self.assertFalse(request_satisfies({**stored, 'multipv': 2}, result, {**requested, 'multipv': 2}))
        self.assertFalse(request_satisfies(stored, result, evaluation()))

    def test_maia_requires_same_ratings_model_and_device(self):
        stored = maia_request(MAIA, 1600, 1600)
        result = {'policy': {'e2e4': 1.}, 'value': .5}
        self.assertTrue(request_satisfies(stored, result, deepcopy(stored)))
        for change in ({'own_rating': 1800}, {'opponent_rating': 1800}, {'version': 2},
                       {'engine': {**MAIA, 'device': 'cuda'}}, {'engine': {}}, {'future_option': True}):
            self.assertFalse(request_satisfies(stored, result, {**stored, **change}))
        self.assertFalse(request_satisfies({'future': 1}, {}, {'future': 1}))

    def test_promotion_joins_both_limits_and_all_candidates_without_scores(self):
        requested = evaluation(18, 10, ('e2e4',))
        records = [
            {'request': evaluation(28, 2, ('d2d4', 'g1f3')), 'result': frame()},
            {'request': evaluation(22, 20, ('c2c4', 'b1c3')), 'result': frame()},
        ]
        original = deepcopy((requested, records))
        promoted = promote_request(records, requested)
        self.assertEqual(promoted['depth'], 28)
        self.assertEqual(promoted['budget_seconds'], 20)
        self.assertEqual(promoted['max_budget_seconds'], 20)
        self.assertEqual(promoted['candidate_moves'], ['e2e4', 'd2d4', 'g1f3', 'c2c4', 'b1c3'])
        self.assertEqual((requested, records), original)
        self.assertEqual(promoted, promote_request(list(reversed(records)), requested))
        self.assertTrue(request_satisfies(promoted, frame(), requested))

    def test_promotion_excludes_incompatible_or_incomplete_observations(self):
        requested = evaluation()
        bad = [
            {'request': {**evaluation(40, 100), 'engine': {**ENGINE, 'threads': 8}}, 'result': frame()},
            {'request': {**evaluation(40, 100), 'policy_version': 1}, 'result': frame()},
            {'request': evaluation(40, 100), 'result': {**frame(), 'coverage_complete': False}},
            {'request': evaluation(40, 100, strategy='staged'), 'result': frame(strategy='staged')},
        ]
        self.assertEqual(promote_request(bad, requested), requested)

    def test_exploration_promotion_never_combines_restricted_searches(self):
        requested = stockfish_exploration_request(ENGINE, 2000, 18, 1, None)
        result = {'lines': [{'cp': 15, 'mate': None, 'depth': 1, 'pv_uci': ['e2e4']}]}
        records = [{'request': {**requested, 'depth': 24, 'movetime_ms': 1000}, 'result': result},
                   {'request': {**requested, 'root_moves': ['e2e4'], 'movetime_ms': 9000}, 'result': result}]
        self.assertEqual(promote_request(records, requested), {**requested, 'depth': 24})

    def test_quality_and_dominance_use_requested_not_achieved_depth(self):
        better, worse = evaluation(22, 10), evaluation(18, 5)
        self.assertGreater(quality_key(better, frame()), quality_key(worse, frame(depth=40)))
        self.assertTrue(dominates(better, frame(), worse, frame(depth=40)))
        self.assertFalse(dominates(worse, frame(depth=40), better, frame()))
        self.assertFalse(dominates(evaluation(28, 1), frame(depth=40), better, frame()))


class HistoryPolicyTests(unittest.TestCase):
    def setUp(self):
        self.first = board('g1f3 g8f6 b1c3 b8c6')
        self.second = board('b1c3 b8c6 g1f3 g8f6')
        self.assertEqual(self.first.fen(), self.second.fen())

    def test_maia_uses_only_matching_recent_input_window(self):
        request = maia_request(MAIA, 1600, 1600)
        self.assertFalse(history_compatible(self.first, self.second, 'maia', request))
        for current in (self.first, self.second):
            current.push_uci('d2d4')
            current.push_uci('d7d5')
        self.assertTrue(history_compatible(self.first, self.second, 'maia', request))
        longer = maia_request({**MAIA, 'history_window': 4}, 1600, 1600)
        self.assertFalse(history_compatible(self.first, self.second, 'maia', longer))

    def test_stockfish_preserves_reversible_history_but_drops_irreversible_prefix(self):
        request = evaluation()
        self.assertFalse(history_compatible(self.first, self.second, 'stockfish', request))
        for current in (self.first, self.second):
            current.push_uci('d2d4')
        self.assertTrue(history_compatible(self.first, self.second, 'stockfish', request))
        self.assertTrue(history_compatible(self.first, chess.Board(self.first.fen()), 'stockfish', request))

    def test_fen_only_import_does_not_erase_live_repetition_context(self):
        repeated = board('g1f3 g8f6 f3g1 f6g8')
        self.assertFalse(history_compatible(repeated, chess.Board(repeated.fen()), 'stockfish', evaluation()))

    def test_maia_does_not_reuse_nonterminal_policy_for_history_draw(self):
        repeated = board(' '.join(['g1f3 g8f6 f3g1 f6g8'] * 4))
        position = chess.Board(repeated.fen())
        self.assertTrue(repeated.is_game_over())
        self.assertFalse(position.is_game_over())
        request = maia_request({**MAIA, 'history_window': 1}, 1600, 1600)
        self.assertFalse(history_compatible(repeated, position, 'maia', request))

    def test_unknown_or_missing_history_model_information_is_exact_only(self):
        requests = [('future', {}), ('maia', maia_request({'maia': 'sha256:fixture'}, 1600, 1600)),
                    ('stockfish', {**evaluation(), 'engine': {}})]
        for namespace, request in requests:
            self.assertTrue(history_compatible(self.first, self.first.copy(stack=True), namespace, request))
            self.assertFalse(history_compatible(self.first, self.second, namespace, request))

    def test_different_full_fen_is_never_compatible(self):
        changed = self.first.copy(stack=True)
        changed.halfmove_clock += 1
        self.assertFalse(history_compatible(changed, self.first, 'stockfish', evaluation()))


if __name__ == '__main__':
    unittest.main()
