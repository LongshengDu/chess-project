from engine.settings import CONFIG as ENGINE_CONFIG
import unittest
from collections import OrderedDict, deque
import threading
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import chess
import torch
from maia3.utils import mirror_move
from maia3.dataset import tokenize_board

from engine.maia import MaiaPolicy


class MaiaBatchTests(unittest.TestCase):
    def test_asset_signature_requires_neither_cuda_nor_engine_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model.pt'
            checkpoint.write_bytes(b'fixture')
            with patch('engine.maia.torch.cuda.is_available', return_value=False), \
                 patch('engine.maia.Maia3UCIEngine') as engine:
                signature = MaiaPolicy.asset_signature('maia3-79m', Path(directory), 'cuda', checkpoint=checkpoint)
            self.assertEqual(signature['device'], 'cuda')
            self.assertTrue(signature['maia'].startswith('sha256:'))
            engine.assert_not_called()

    def test_asset_signature_never_discovers_remote_checkpoint_names(self):
        config = SimpleNamespace(checkpoint_path=None, checkpoint_filename=None,
                                 model_spec=SimpleNamespace(checkpoint_filename=None))
        with patch('engine.maia.resolve_checkpoint_path') as resolver:
            with self.assertRaises(FileNotFoundError):
                MaiaPolicy._signature(config)
        resolver.assert_not_called()

    def test_signature_resolves_checkpoint_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'model.pt'
            checkpoint.write_bytes(b'fixture')
            config = SimpleNamespace(checkpoint_path=None, history=8, device='cpu',
                model_spec='fixture', checkpoint_filename='model.pt', cache_dir=directory, revision=None)
            engine = Mock(cfg=config)
            policy = MaiaPolicy.__new__(MaiaPolicy)
            policy._engine = engine
            policy._inference_lock = threading.RLock()
            with patch('engine.maia.resolve_checkpoint_path', return_value=str(checkpoint)) as resolve:
                first = policy.model_signature
                self.assertEqual(first, policy.model_signature)
            from engine.assets_identity import asset_identity
            self.assertEqual(first['maia'], asset_identity(checkpoint))
            resolve.assert_called_once()
            self.assertTrue(resolve.call_args.kwargs['local_files_only'])
            engine.ensure_model_loaded.assert_not_called()

    def test_history_tail_is_exactly_equivalent_to_full_replay(self):
        board=chess.Board()
        for move in ('e2e4','c7c5','g1f3','b8c6','f1b5','g8f6','e1g1','a7a6','b5c6','d7c6'):
            board.push_uci(move)
        for length in (1,3,8,20):
            walk=board.root(); expected=deque([tokenize_board(walk)],maxlen=length)
            for move in board.move_stack:
                walk.push(move); expected.append(tokenize_board(walk))
            with patch('engine.maia.tokenize_board',wraps=tokenize_board) as tokenize:
                actual=MaiaPolicy._history(board,length)
            torch.testing.assert_close(torch.stack(list(actual)),torch.stack(list(expected)))
            self.assertEqual(tokenize.call_count,min(length,len(board.move_stack)+1))

    def test_history_sensitive_cache_and_chunked_batches(self):
        a,b=chess.Board(),chess.Board()
        for move in ('g1f3','g8f6','b1c3','b8c6'):a.push_uci(move)
        for move in ('b1c3','b8c6','g1f3','g8f6'):b.push_uci(move)
        self.assertEqual(a.fen(),b.fen())
        mapping={m.uci():i for i,m in enumerate(a.legal_moves)}
        builder=Mock(return_value=torch.zeros(64,12))
        model=Mock(side_effect=lambda tokens,own,other:(torch.zeros(len(own),len(mapping)),torch.zeros(len(own),3),None))
        policy=MaiaPolicy.__new__(MaiaPolicy)
        policy._inference_lock = threading.RLock()
        policy._prepared_cache = OrderedDict()
        policy._engine=SimpleNamespace(ensure_model_loaded=lambda:None,all_moves_dict=mapping,
            _tokens_from_history=builder,model=model,cfg=SimpleNamespace(device='cpu',use_amp=False,history=3))
        boards=[a,b,a,b,a]
        with patch.dict(ENGINE_CONFIG['MAIA'], BATCH_SIZE=2):
            result=policy.batch_evaluate([x.fen() for x in boards],[1500]*5,[1500]*5,boards=boards)
        self.assertEqual(model.call_count,3)
        self.assertEqual(builder.call_count,2)
        self.assertEqual(len(result),5)
        self.assertFalse(torch.equal(torch.stack(list(builder.call_args_list[0].args[0])),
                                     torch.stack(list(builder.call_args_list[1].args[0]))))
        for row in result:self.assertAlmostEqual(sum(row['policy'].values()),1.,places=6)
        # Single-position calls must preserve the same legal policy/history.
        single = {move.uci(): probability for move, probability in policy.probabilities(a)}
        self.assertEqual(single, result[0]['policy'])
        self.assertEqual(builder.call_count, 2)
        with self.assertRaisesRegex(ValueError, 'one history board'):
            policy.batch_evaluate([a.fen(), b.fen()], [1500]*2, [1500]*2, boards=[a])

    def test_mixed_turn_batches_keep_legal_normalized_policy_and_white_value(self):
        white = chess.Board()
        black = chess.Board()
        black.push_uci('e2e4')
        mate = chess.Board('7k/6Q1/6K1/8/8/8/8/8 b - - 0 1')
        boards = [white, white, black, mate]
        moves = dict.fromkeys(move.uci() if board.turn else mirror_move(move.uci())
                              for board in boards for move in board.legal_moves)
        mapping = {move: index for index, move in enumerate(moves)}
        token_builder = Mock(return_value=torch.zeros(64, 12))
        logits = torch.arange(len(mapping), dtype=torch.float32).repeat(4, 1) / 10
        values = torch.tensor([[0., 0., 1.]]).repeat(4, 1)
        policy = MaiaPolicy.__new__(MaiaPolicy)
        policy._inference_lock = threading.RLock()
        policy._prepared_cache = OrderedDict()
        policy._engine = SimpleNamespace(
            ensure_model_loaded=lambda: None, all_moves_dict=mapping,
            _tokens_from_history=token_builder,
            cfg=SimpleNamespace(device='cpu', use_amp=False, history=1),
            model=Mock(return_value=(logits, values, None)),
        )
        result = policy.batch_evaluate([board.fen() for board in boards], [600, 1500, 2600, 1500], [1500]*4)
        self.assertEqual(token_builder.call_count, 3)
        self.assertEqual(result[0]['policy'], result[1]['policy'])
        for board, evaluation in zip(boards[:3], result[:3]):
            self.assertEqual(set(evaluation['policy']), {move.uci() for move in board.legal_moves})
            self.assertAlmostEqual(sum(evaluation['policy'].values()), 1., places=6)
            self.assertEqual(list(evaluation['policy'].values()), sorted(evaluation['policy'].values(), reverse=True))
        self.assertAlmostEqual(result[0]['value'] + result[2]['value'], 1.)
        self.assertEqual(result[3], {'policy':{}, 'value':1.})
        policy._engine.cfg.history = 8
        token_builder.reset_mock()
        policy.batch_evaluate([board.fen() for board in boards], [600,1500,2600,1500], [1500]*4,
                              boards={board.fen():board for board in boards})
        self.assertEqual([len(call.args[0]) for call in token_builder.call_args_list], [1,2,1])
        with self.assertRaises(ValueError):
            policy.batch_evaluate([white.fen()], [1500], [1500], boards={white.fen():black})


if __name__ == '__main__':
    unittest.main()
