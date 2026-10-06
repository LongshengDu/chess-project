"""Offline Node bootstrap, dependencies and build orchestration regressions."""
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

from web import build
from web.build_node import NodeRuntime
from web.build_packages import FrontendPackages


class BuildTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.runtime = NodeRuntime({'TOOLS_DIR': self.directory, 'NODE_MIN_VERSION': 22,
                                    'NODE_DIST_URL': 'https://example.invalid', 'NODE_EXECUTABLE': None})

    def test_failed_or_hanging_version_probe_is_not_an_usable_node(self):
        with patch('web.build_node.subprocess.run', return_value=Mock(returncode=1, stdout='v22.0.0')):
            self.assertIsNone(self.runtime.node_version(Path('unused')))
        with patch('web.build_node.subprocess.run', side_effect=subprocess.TimeoutExpired('node', 10)):
            self.assertIsNone(self.runtime.node_version(Path('unused')))

    def test_invalid_checksum_manifest_cannot_choose_an_outside_download(self):
        for filename, digest in (('../../node-v22.0.0-win-x64.zip', '0'*64),
                                 ('node-v22.0.0-win-x64.zip', 'invalid')):
            with self.subTest(filename=filename, digest=digest), \
                 patch.object(self.runtime, 'node_archive_target', return_value=('win-x64', 'zip')), \
                 patch.object(self.runtime, 'read_url', return_value=f'{digest}  {filename}\n'.encode()), \
                 patch.object(self.runtime, 'download') as download:
                with self.assertRaisesRegex(SystemExit, 'invalid archive'):
                    self.runtime.install_node()
                download.assert_not_called()

    def test_archive_traversal_preserves_existing_installation(self):
        archive = self.directory / 'fixture.zip'
        destination = self.directory / 'node-v22.0.0-win-x64'
        destination.mkdir()
        existing = destination / 'node.exe'
        existing.write_bytes(b'old node')
        with zipfile.ZipFile(archive, 'w') as package:
            package.writestr('../outside.txt', b'unsafe')
        with self.assertRaisesRegex(SystemExit, 'Unsafe path'):
            self.runtime.extract_node(archive, destination, destination.name)
        self.assertEqual(existing.read_bytes(), b'old node')
        with self.assertRaisesRegex(SystemExit, 'must stay inside'):
            self.runtime.extract_node(archive, self.directory.parent / 'outside', destination.name)

    def test_tar_special_file_is_rejected_without_replacing_installation(self):
        archive = self.directory / 'fixture.tar.xz'
        with tarfile.open(archive, 'w:xz') as package:
            entry = tarfile.TarInfo('node/fifo')
            entry.type = tarfile.FIFOTYPE
            package.addfile(entry)
        with self.assertRaisesRegex(SystemExit, 'Unsupported archive entry'):
            self.runtime.extract_node(archive, self.directory / 'node', 'node')

    def test_missing_dependencies_include_changed_pins_and_malformed_manifests(self):
        packages = FrontendPackages(self.directory)
        package = {'dependencies': {'one': '1.0.0', '@scope/two': '2.0.0', 'three': '3.0.0'}}
        for name, text in [('one', '{"version":"1.0.0"}'), ('@scope/two', '{"version":"1.9.0"}'), ('three', '{')]:
            manifest = self.directory / 'node_modules' / name / 'package.json'
            manifest.parent.mkdir(parents=True)
            manifest.write_text(text, encoding='utf-8')
        self.assertEqual(packages.missing_dependencies(package), ['@scope/two', 'three'])

    def test_build_reuses_runtime_and_runs_typecheck_before_bundling(self):
        with patch.object(build, 'NodeRuntime') as runtime, \
             patch.object(build, 'FrontendPackages') as packages, \
             patch.object(Path, 'is_dir', return_value=True), \
             patch.object(build.subprocess, 'run') as run:
            runtime.return_value.find_node.return_value = (Path('fixture/node'), 'v22.0.0')
            packages.return_value.missing_dependencies.return_value = []
            build.main()
        runtime.return_value.install_node.assert_not_called()
        packages.return_value.install_dependencies.assert_not_called()
        self.assertEqual([call.args[0][-1] for call in run.call_args_list], ['--noEmit', 'build'])
        for call in run.call_args_list:
            self.assertTrue(call.kwargs['check'])
            self.assertEqual(call.kwargs['cwd'], build.FRONTEND)


if __name__ == '__main__':
    unittest.main()
