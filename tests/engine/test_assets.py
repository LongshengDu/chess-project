"""Engine asset validation using temporary archives and mocked downloads."""
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from engine import assets_maia, assets_stockfish
from engine.settings import CONFIG


class AssetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_checkpoint_spec_is_resolved_from_current_configuration(self):
        checkpoint = self.directory / 'current.pt'
        checkpoint.write_bytes(b'fixture')
        with patch.dict(CONFIG['MAIA'], MODEL='current-model', CHECKPOINT=None, CACHE_DIR=self.directory), \
             patch.object(assets_maia, 'resolve_model_spec', return_value=SimpleNamespace(checkpoint_filename='current.pt')) as resolve, \
             patch.object(assets_maia.subprocess, 'run') as download:
            assets_maia.ensure_model_cached()
        resolve.assert_called_once_with('current-model')
        download.assert_not_called()

    def test_empty_checkpoint_is_not_treated_as_a_cached_model(self):
        (self.directory / 'empty.pt').touch()
        with patch.dict(CONFIG['MAIA'], MODEL='fixture', CHECKPOINT=None, CACHE_DIR=self.directory), \
             patch.object(assets_maia, 'resolve_model_spec', return_value=SimpleNamespace(checkpoint_filename='empty.pt')), \
             patch.object(assets_maia.subprocess, 'run') as download:
            assets_maia.ensure_model_cached()
        download.assert_called_once()

    def test_verified_stockfish_download_checks_both_size_and_digest(self):
        body = b'stockfish archive fixture'
        asset = {'name': 'stockfish.zip', 'browser_download_url': 'https://example.invalid/stockfish.zip',
                 'size': len(body), 'digest': 'sha256:' + hashlib.sha256(body).hexdigest()}
        with patch.dict(CONFIG['STOCKFISH'], CACHE_DIR=self.directory):
            for overrides in ({}, {'size': len(body) + 1}, {'digest': 'sha256:' + '0'*64}):
                with self.subTest(overrides=overrides), \
                     patch.object(assets_stockfish.urllib.request, 'urlopen', return_value=io.BytesIO(body)):
                    if overrides:
                        with self.assertRaisesRegex(RuntimeError, 'verification'):
                            assets_stockfish._download_stockfish_archive({**asset, **overrides})
                    else:
                        archive = assets_stockfish._download_stockfish_archive(asset)
                        self.assertEqual(archive.read_bytes(), body)
                        archive.unlink()
                    self.assertEqual(list(self.directory.iterdir()), [])

    def test_archive_copies_only_unique_executable_without_extracting_paths(self):
        archive = self.directory / 'fixture.zip'
        output = self.directory / 'stockfish.exe'
        with zipfile.ZipFile(archive, 'w') as package:
            package.writestr('../../unrelated.txt', b'not extracted')
            package.writestr('stockfish/stockfish.exe', b'engine')
        with patch.dict(CONFIG['STOCKFISH'], CACHE_DIR=self.directory):
            assets_stockfish._extract_stockfish(archive, archive.name, output.name, output)
            self.assertEqual(output.read_bytes(), b'engine')
            with zipfile.ZipFile(archive, 'a') as package:
                package.writestr('duplicate/stockfish.exe', b'other engine')
            with self.assertRaisesRegex(RuntimeError, 'did not contain'):
                assets_stockfish._extract_stockfish(archive, archive.name, output.name, output)
        self.assertEqual(output.read_bytes(), b'engine')
        self.assertEqual({p.name for p in self.directory.iterdir()}, {'fixture.zip', 'stockfish.exe'})


if __name__ == '__main__':
    unittest.main()
