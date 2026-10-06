"""Discover or install a checksum-verified Node.js runtime for frontend builds."""
from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from .settings import CONFIG


class NodeRuntime:
    def __init__(self, config=None):
        self.config = CONFIG['FRONTEND']['BUILD'] if config is None else config
        self.tools = self.config['TOOLS_DIR']
        self.minimum_version = self.config['NODE_MIN_VERSION']
        self.distribution_url = self.config['NODE_DIST_URL']

    @staticmethod
    def node_version(executable: Path) -> tuple[int, str] | None:
        try:
            result = subprocess.run(
                [str(executable), "--version"], capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode:
            return None
        match = re.fullmatch(r"v(\d+).*", result.stdout.strip())
        return (int(match.group(1)), result.stdout.strip()) if match else None

    def find_node(self) -> tuple[Path, str] | None:
        if self.config["NODE_EXECUTABLE"]:
            candidate = Path(shutil.which(self.config["NODE_EXECUTABLE"]) or self.config["NODE_EXECUTABLE"]).resolve()
            version = self.node_version(candidate)
            if version is None or version[0] < self.minimum_version:
                raise SystemExit(f'FRONTEND.BUILD.NODE_EXECUTABLE in config.yaml must point to Node.js {self.minimum_version}+: {candidate}')
            return candidate, version[1]
        local_pattern = "node-v*-win-*/node.exe" if os.name == "nt" else "node-v*-*/bin/node"
        candidates = [
            Path(path)
            for path in [shutil.which("node")]
            if path
        ]
        candidates += sorted(
            self.tools.glob(local_pattern),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if os.name == "nt" and os.environ.get("APPDATA"):
            cache = Path(os.environ["APPDATA"]) / "npm-cache" / "_npx"
            candidates += sorted(
                cache.glob("*/node_modules/node/bin/node.exe"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )

        found = []
        for candidate in dict.fromkeys(path.resolve() for path in candidates):
            version = self.node_version(candidate)
            if version:
                found.append(version[1])
                if version[0] >= self.minimum_version:
                    return candidate, version[1]
        if found:
            print(
                f"Node.js {self.minimum_version}+ is required; found only {', '.join(found)}.",
                flush=True,
            )
        return None

    def node_archive_target(self) -> tuple[str, str]:
        systems = {"windows": "win", "linux": "linux", "darwin": "darwin"}
        architectures = {
            "amd64": "x64",
            "x86_64": "x64",
            "arm64": "arm64",
            "aarch64": "arm64",
        }
        system = systems.get(platform.system().lower())
        architecture = architectures.get(platform.machine().lower())
        if not system or not architecture:
            raise SystemExit(
                "Automatic Node.js installation is not available for "
                f"{platform.system()} {platform.machine()}. Install Node.js {self.minimum_version}+ manually "
                "or set FRONTEND.BUILD.NODE_EXECUTABLE in config.yaml to its executable."
            )
        extension = "zip" if system == "win" else "tar.xz"
        return f"{system}-{architecture}", extension

    @staticmethod
    def read_url(url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": "chess-project-build"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except (OSError, urllib.error.URLError) as error:
            raise SystemExit(f"Could not download {url}: {error}") from error

    @staticmethod
    def download(url: str, destination: Path, expected_sha256: str) -> None:
        if destination.exists():
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if digest == expected_sha256:
                return
            destination.unlink()

        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "chess-project-build"})
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                digest = hashlib.sha256()
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
        except (OSError, urllib.error.URLError) as error:
            temporary.unlink(missing_ok=True)
            raise SystemExit(f"Could not download {url}: {error}") from error

        if digest.hexdigest() != expected_sha256:
            temporary.unlink(missing_ok=True)
            raise SystemExit(f"Checksum verification failed for {url}.")
        temporary.replace(destination)

    @staticmethod
    def is_within(directory: Path, target: Path) -> bool:
        try:
            target.resolve().relative_to(directory.resolve())
        except ValueError:
            return False
        return True

    def extract_node(self, archive: Path, destination: Path, directory_name: str) -> Path:
        # Validate the final destination before replacing an existing Node install.
        if destination.resolve() == self.tools.resolve() or not self.is_within(self.tools, destination):
            raise SystemExit(f"Node.js installation must stay inside {self.tools}.")
        with tempfile.TemporaryDirectory(prefix="node-extract-", dir=self.tools) as temporary:
            extract_to = Path(temporary)
            if archive.suffix == ".zip":
                with zipfile.ZipFile(archive) as package:
                    if any(not self.is_within(extract_to, extract_to / item.filename) for item in package.infolist()):
                        raise SystemExit(f"Unsafe path found in {archive.name}.")
                    package.extractall(extract_to)
            else:
                with tarfile.open(archive, mode="r:xz") as package:
                    for item in package.getmembers():
                        item_path = extract_to / item.name
                        if not self.is_within(extract_to, item_path):
                            raise SystemExit(f"Unsafe path found in {archive.name}.")
                        if item.issym() and not self.is_within(extract_to, item_path.parent / item.linkname):
                            raise SystemExit(f"Unsafe link found in {archive.name}.")
                        if item.islnk() and not self.is_within(extract_to, extract_to / item.linkname):
                            raise SystemExit(f"Unsafe link found in {archive.name}.")
                        if not (item.isfile() or item.isdir() or item.issym() or item.islnk()):
                            raise SystemExit(f"Unsupported archive entry in {archive.name}.")
                        # Resolve against links already extracted, so a chain of
                        # individually relative links cannot escape the directory.
                        package.extract(item, extract_to)

            extracted = extract_to / directory_name
            if extracted.resolve() == extract_to.resolve() or not self.is_within(extract_to, extracted):
                raise SystemExit(f"Unsafe directory name in {archive.name}.")
            if not extracted.is_dir():
                raise SystemExit(f"The Node.js archive did not contain {directory_name}.")
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(extracted), destination)
        return destination / ("node.exe" if os.name == "nt" else "bin/node")

    def install_node(self) -> tuple[Path, str]:
        target, extension = self.node_archive_target()
        checksums_url = f"{self.distribution_url}/latest-v{self.minimum_version}.x/SHASUMS256.txt"
        checksums = self.read_url(checksums_url).decode("utf-8")
        suffix = f"-{target}.{extension}"
        match = next(
            (
                (parts[0], parts[1])
                for line in checksums.splitlines()
                if len(parts := line.split()) == 2 and parts[1].endswith(suffix)
            ),
            None,
        )
        if not match:
            raise SystemExit(f"No Node.js {self.minimum_version} archive is available for {target}.")

        checksum, filename = match
        if (not re.fullmatch(r"[0-9a-fA-F]{64}", checksum)
                or not re.fullmatch(r"node-v\d+\.\d+\.\d+-" + re.escape(target + '.' + extension), filename)):
            raise SystemExit('The Node.js checksum manifest contains an invalid archive name or digest.')
        directory_name = filename.removesuffix(f".{extension}")
        install_directory = self.tools / directory_name
        executable = install_directory / ("node.exe" if os.name == "nt" else "bin/node")
        installed_version = self.node_version(executable)
        if installed_version and installed_version[0] >= self.minimum_version:
            return executable, installed_version[1]

        self.tools.mkdir(parents=True, exist_ok=True)
        archive = self.tools / "downloads" / filename
        self.download(f"{self.distribution_url}/latest-v{self.minimum_version}.x/{filename}", archive, checksum)
        print(f"Installing Node.js in {install_directory}", flush=True)
        executable = self.extract_node(archive, install_directory, directory_name)
        installed_version = self.node_version(executable)
        if not installed_version or installed_version[0] < self.minimum_version:
            raise SystemExit("The automatically installed Node.js executable is invalid.")
        return executable, installed_version[1]
