"""Backend startup and engine ownership without opening a server."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app import main
from backend.settings import CONFIG


class AppTests(unittest.TestCase):
    def test_default_launch_uses_one_shared_model_and_closes_stockfish(self):
        with tempfile.TemporaryDirectory() as temp:
            dist = Path(temp) / 'web' / 'dist'
            dist.mkdir(parents=True)
            (dist / 'index.html').write_text('web')
            with (
                patch.dict(CONFIG['FRONTEND'], STATIC_DIR=dist),
                patch('backend.app.ensure_runtime_assets', return_value=Path('stockfish')),
                patch('backend.app.MaiaPolicy') as maia,
                patch('backend.app.StockfishScorer') as stockfish,
                patch('backend.app.PlatformAnalysis') as platform,
                patch('backend.app.create_app') as create,
            ):
                main(['--port', '5050'])
                maia.assert_called_once()
                stockfish.assert_called_once()
                self.assertEqual(platform.call_args.args[:2], (maia.return_value, stockfish.return_value))
                create.assert_called_once_with(platform.return_value, dist)
                create.return_value.run.assert_called_once_with(host='127.0.0.1', port=5050, threaded=True)
                stockfish.return_value.close.assert_called_once()

    def test_startup_failure_closes_stockfish(self):
        with tempfile.TemporaryDirectory() as temp:
            dist = Path(temp)
            (dist / 'index.html').write_text('web')
            with (
                patch.dict(CONFIG['FRONTEND'], STATIC_DIR=dist),
                patch('backend.app.ensure_runtime_assets', return_value=Path('stockfish')),
                patch('backend.app.MaiaPolicy'),
                patch('backend.app.StockfishScorer') as stockfish,
                patch('backend.app.PlatformAnalysis', side_effect=RuntimeError('database unavailable')),
                patch('backend.app.create_app') as create,
                self.assertRaisesRegex(RuntimeError, 'database unavailable'),
            ):
                try:
                    main([])
                finally:
                    stockfish.return_value.close.assert_called_once()
                    create.assert_not_called()


if __name__ == '__main__':
    unittest.main()
