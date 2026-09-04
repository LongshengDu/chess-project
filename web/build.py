from __future__ import annotations

import hashlib
import json
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


WEB = Path(__file__).resolve().parent
TOOLS = WEB / ".tools"
MIN_NODE = 22
NODE_DIST_URL = "https://nodejs.org/dist"


def node_version(executable: Path) -> tuple[int, str] | None:
    try:
        result = subprocess.run(
            [str(executable), "--version"], capture_output=True, text=True
        )
    except OSError:
        return None
    match = re.fullmatch(r"v(\d+).*", result.stdout.strip())
    return (int(match.group(1)), result.stdout.strip()) if match else None


def find_node() -> tuple[Path, str] | None:
    local_pattern = "node-v*-win-*/node.exe" if os.name == "nt" else "node-v*-*/bin/node"
    candidates = [
        Path(path)
        for path in [os.environ.get("NODE"), shutil.which("node")]
        if path
    ]
    candidates += sorted(
        TOOLS.glob(local_pattern),
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
        version = node_version(candidate)
        if version:
            found.append(version[1])
            if version[0] >= MIN_NODE:
                return candidate, version[1]
    if found:
        print(
            f"Node.js {MIN_NODE}+ is required; found only {', '.join(found)}.",
            flush=True,
        )
    return None


def node_archive_target() -> tuple[str, str]:
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
            f"{platform.system()} {platform.machine()}. Install Node.js {MIN_NODE}+ manually "
            "or set NODE to its executable."
        )
    extension = "zip" if system == "win" else "tar.xz"
    return f"{system}-{architecture}", extension


def read_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "chess-project-build"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except (OSError, urllib.error.URLError) as error:
        raise SystemExit(f"Could not download {url}: {error}") from error


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


def is_within(directory: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def extract_node(archive: Path, destination: Path, directory_name: str) -> Path:
    with tempfile.TemporaryDirectory(prefix="node-extract-", dir=TOOLS) as temporary:
        extract_to = Path(temporary)
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as package:
                if any(not is_within(extract_to, extract_to / item.filename) for item in package.infolist()):
                    raise SystemExit(f"Unsafe path found in {archive.name}.")
                package.extractall(extract_to)
        else:
            with tarfile.open(archive, mode="r:xz") as package:
                for item in package.getmembers():
                    item_path = extract_to / item.name
                    if not is_within(extract_to, item_path):
                        raise SystemExit(f"Unsafe path found in {archive.name}.")
                    if item.issym() and not is_within(extract_to, item_path.parent / item.linkname):
                        raise SystemExit(f"Unsafe link found in {archive.name}.")
                    if item.islnk() and not is_within(extract_to, extract_to / item.linkname):
                        raise SystemExit(f"Unsafe link found in {archive.name}.")
                package.extractall(extract_to)

        extracted = extract_to / directory_name
        if not extracted.is_dir():
            raise SystemExit(f"The Node.js archive did not contain {directory_name}.")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(extracted), destination)
    return destination / ("node.exe" if os.name == "nt" else "bin/node")


def install_node() -> tuple[Path, str]:
    target, extension = node_archive_target()
    checksums_url = f"{NODE_DIST_URL}/latest-v{MIN_NODE}.x/SHASUMS256.txt"
    checksums = read_url(checksums_url).decode("utf-8")
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
        raise SystemExit(f"No Node.js {MIN_NODE} archive is available for {target}.")

    checksum, filename = match
    directory_name = filename.removesuffix(f".{extension}")
    install_directory = TOOLS / directory_name
    executable = install_directory / ("node.exe" if os.name == "nt" else "bin/node")
    installed_version = node_version(executable)
    if installed_version and installed_version[0] >= MIN_NODE:
        return executable, installed_version[1]

    TOOLS.mkdir(parents=True, exist_ok=True)
    archive = TOOLS / "downloads" / filename
    download(f"{NODE_DIST_URL}/latest-v{MIN_NODE}.x/{filename}", archive, checksum)
    print(f"Installing Node.js in {install_directory}", flush=True)
    executable = extract_node(archive, install_directory, directory_name)
    installed_version = node_version(executable)
    if not installed_version or installed_version[0] < MIN_NODE:
        raise SystemExit("The automatically installed Node.js executable is invalid.")
    return executable, installed_version[1]


def find_npx_cli(node: Path, env: dict[str, str]) -> Path | None:
    candidates = [
        node.parent / "node_modules" / "npm" / "bin" / "npx-cli.js",
        node.parent.parent / "lib" / "node_modules" / "npm" / "bin" / "npx-cli.js",
    ]
    npx = shutil.which("npx", path=env["PATH"])
    if npx:
        resolved = Path(npx).resolve()
        if resolved.suffix == ".js":
            candidates.append(resolved)
        candidates.append(resolved.parent / "node_modules" / "npm" / "bin" / "npx-cli.js")
    return next((path for path in candidates if path.is_file()), None)


def find_corepack_cli(node: Path) -> Path | None:
    candidates = [
        node.parent / "node_modules" / "corepack" / "dist" / "corepack.js",
        node.parent.parent
        / "lib"
        / "node_modules"
        / "corepack"
        / "dist"
        / "corepack.js",
    ]
    return next((path for path in candidates if path.is_file()), None)


def has_package_runner(node: Path, env: dict[str, str]) -> bool:
    return bool(find_corepack_cli(node) or find_npx_cli(node, env))


def package_manifest() -> dict[str, object]:
    return json.loads((WEB / "package.json").read_text(encoding="utf-8"))


def missing_dependencies(package: dict[str, object]) -> list[str]:
    required = {
        **package.get("dependencies", {}),
        **package.get("devDependencies", {}),
    }
    missing = []
    for name, wanted_version in required.items():
        installed_manifest = WEB / "node_modules" / Path(*name.split("/")) / "package.json"
        try:
            installed_version = json.loads(installed_manifest.read_text(encoding="utf-8"))["version"]
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            missing.append(name)
            continue
        if re.fullmatch(r"\d+\.\d+\.\d+(?:[-+].+)?", str(wanted_version)):
            if installed_version != wanted_version:
                missing.append(name)
    return missing


def install_dependencies(
    node: Path, env: dict[str, str], package: dict[str, object]
) -> None:
    package_manager = str(package["packageManager"])
    manager = package_manager.partition("@")[0]
    corepack_cli = find_corepack_cli(node)
    if corepack_cli:
        command = [str(node), str(corepack_cli), manager, "install", "--frozen-lockfile"]
    else:
        npx_cli = find_npx_cli(node, env)
        if not npx_cli:
            raise SystemExit("The Node.js installation does not include Corepack or npm/npx.")
        command = [
            str(node),
            str(npx_cli),
            "--yes",
            "--package",
            package_manager,
            manager,
            "install",
            "--frozen-lockfile",
        ]

    print(f"Installing frontend dependencies with {package_manager}", flush=True)
    subprocess.run(
        command,
        cwd=WEB,
        env={
            **env,
            "CI": env.get("CI") or "true",
            "COREPACK_ENABLE_DOWNLOAD_PROMPT": "0",
            "npm_config_yes": "true",
        },
        check=True,
    )


def main() -> None:
    found_node = find_node()
    if found_node:
        node, version = found_node
    else:
        print(f"Node.js {MIN_NODE}+ was not found; installing it automatically.", flush=True)
        node, version = install_node()
    env = {
        **os.environ,
        "PATH": str(node.parent) + os.pathsep + os.environ.get("PATH", ""),
    }
    package = package_manifest()
    missing = missing_dependencies(package)
    tsc = WEB / "node_modules" / "typescript" / "bin" / "tsc"
    vite = WEB / "node_modules" / "vite" / "bin" / "vite.js"

    print(f"Using Node.js {version}", flush=True)
    if missing:
        if not has_package_runner(node, env):
            node, version = install_node()
            env["PATH"] = str(node.parent) + os.pathsep + os.environ.get("PATH", "")
            print(f"Using Node.js {version} with its bundled package manager", flush=True)
        print(f"Missing or outdated packages: {', '.join(missing)}", flush=True)
        install_dependencies(node, env, package)

    subprocess.run([str(node), str(tsc), "--noEmit"], cwd=WEB, env=env, check=True)
    subprocess.run([str(node), str(vite), "build"], cwd=WEB, env=env, check=True)


if __name__ == "__main__":
    main()
