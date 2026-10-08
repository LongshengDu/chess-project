"""Concurrent evidence publication never shares a partially written temporary file."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from analysis.cache.storage import JsonCache, write_json


class CacheTests(unittest.TestCase):
    def test_concurrent_writers_publish_complete_records(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = JsonCache(directory)
            barrier = threading.Barrier(2)
            local = threading.local()
            replace = Path.replace
            records = [{'writer': n, 'data': [n] * 1000} for n in range(2)]

            def publish(source, destination):
                if not getattr(local, 'ready', False):
                    local.ready = True
                    barrier.wait(timeout=5)
                return replace(source, destination)

            with patch.object(Path, 'replace', publish), ThreadPoolExecutor(2) as executor:
                futures = [executor.submit(cache.put, ['same-position'], value) for value in records]
                for future in futures:
                    future.result(timeout=10)
            self.assertIn(cache.get(['same-position']), records)
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_publication_retries_transient_permission_error(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'evidence.json'
            replace = Path.replace
            attempts = []

            def publish(source, destination):
                attempts.append(source)
                if len(attempts) == 1:
                    raise PermissionError('another writer is publishing')
                return replace(source, destination)

            with patch.object(Path, 'replace', publish), patch('analysis.cache.storage.time.sleep') as sleep:
                write_json(target, {'complete': True})
            self.assertEqual(json.loads(target.read_text(encoding='utf-8')), {'complete': True})
            self.assertEqual(len(attempts), 2)
            sleep.assert_called_once_with(0.01)
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_failed_publication_preserves_previous_value_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'evidence.json'
            write_json(target, {'previous': True})
            with patch.object(Path, 'replace', side_effect=OSError('publication failed')):
                with self.assertRaisesRegex(OSError, 'publication failed'):
                    write_json(target, {'next': True})
            self.assertEqual(json.loads(target.read_text(encoding='utf-8')), {'previous': True})
            self.assertEqual(list(Path(directory).iterdir()), [target])
            with self.assertRaises(ValueError):
                write_json(target, {'invalid': float('nan')})
            self.assertEqual(json.loads(target.read_text(encoding='utf-8')), {'previous': True})


if __name__ == '__main__':
    unittest.main()
