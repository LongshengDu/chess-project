"""Asset identity follows bytes rather than installation location or timestamps."""
import os
from pathlib import Path
import tempfile
import unittest

from engine.assets_identity import asset_identity


class AssetIdentityTests(unittest.TestCase):
    def test_equal_bytes_share_identity_after_copy_or_touch(self):
        with tempfile.TemporaryDirectory() as directory:
            original, copied = Path(directory) / 'original', Path(directory) / 'copied'
            original.write_bytes(b'engine fixture')
            copied.write_bytes(original.read_bytes())
            expected = asset_identity(original)
            self.assertEqual(expected, asset_identity(copied))
            os.utime(original, (1_000_000_000, 1_000_000_000))
            self.assertEqual(expected, asset_identity(original))

    def test_changed_bytes_invalidate_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model'
            path.write_bytes(b'first')
            first = asset_identity(path)
            path.write_bytes(b'other')
            self.assertNotEqual(first, asset_identity(path))


if __name__ == '__main__':
    unittest.main()
