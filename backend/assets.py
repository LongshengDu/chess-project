from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from maia3.model_registry import resolve_model_spec

if __package__:
    from .settings import (
        MAIA_CACHE_COMMAND,
        MAIA_CACHE_DIR,
        MAIA_MODEL_NAME,
        STOCKFISH_ASSETS,
        STOCKFISH_CACHE_DIR,
        STOCKFISH_RELEASE_API,
    )
else:  # Support running this file directly.
    from settings import (
        MAIA_CACHE_COMMAND,
        MAIA_CACHE_DIR,
        MAIA_MODEL_NAME,
        STOCKFISH_ASSETS,
        STOCKFISH_CACHE_DIR,
        STOCKFISH_RELEASE_API,
    )


MAIA_MODEL_SPEC = resolve_model_spec(MAIA_MODEL_NAME)


def ensure_model_cached() -> None:
    MAIA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if MAIA_MODEL_SPEC.checkpoint_filename and any(
        MAIA_CACHE_DIR.rglob(MAIA_MODEL_SPEC.checkpoint_filename)
    ):
        return
    subprocess.run(MAIA_CACHE_COMMAND, check=True)


def _stockfish_target() -> tuple[str, str, Path]:
    architectures = {
        "amd64": "x86-64",
        "x86_64": "x86-64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }
    system = platform.system().lower()
    architecture = architectures.get(platform.machine().lower())
    asset_name = STOCKFISH_ASSETS.get((system, architecture or ""))
    if not asset_name:
        raise RuntimeError(
            "No official Stockfish binary is configured for "
            f"{platform.system()} {platform.machine()}"
        )

    archive_suffix = ".tar.gz" if asset_name.endswith(".tar.gz") else Path(asset_name).suffix
    executable_name = asset_name.removesuffix(archive_suffix)
    if system == "windows":
        executable_name += ".exe"
    return asset_name, executable_name, STOCKFISH_CACHE_DIR / executable_name


def _latest_stockfish_asset(asset_name: str) -> tuple[str, dict]:
    request_headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "maia-local-web-chess",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    try:
        request = urllib.request.Request(STOCKFISH_RELEASE_API, headers=request_headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            release = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not read the latest Stockfish release: {error}"
        ) from error

    if not isinstance(release, dict):
        raise TypeError("The latest Stockfish release returned invalid metadata")

    asset = next(
        (
            candidate
            for candidate in release.get("assets", [])
            if isinstance(candidate, dict) and candidate.get("name") == asset_name
        ),
        None,
    )
    if not asset:
        available = ", ".join(
            str(candidate.get("name"))
            for candidate in release.get("assets", [])
            if isinstance(candidate, dict)
        )
        raise RuntimeError(
            f"Stockfish {release.get('tag_name')} has no {asset_name} asset. "
            f"Available assets: {available}"
        )

    tag = release.get("tag_name")
    digest = asset.get("digest")
    url = asset.get("browser_download_url")
    size = asset.get("size")
    expected_url_prefix = (
        "https://github.com/official-stockfish/Stockfish/releases/download/"
    )
    if (
        not tag
        or not isinstance(digest, str)
        or len(digest) != len("sha256:") + 64
        or not digest.startswith("sha256:")
        or not isinstance(url, str)
        or not url.startswith(expected_url_prefix)
        or not isinstance(size, int)
        or size <= 0
    ):
        raise RuntimeError("The latest Stockfish release has incomplete metadata")
    return str(tag), asset


def _download_stockfish_archive(asset: dict) -> Path:
    url = asset["browser_download_url"]
    expected_digest = asset["digest"].removeprefix("sha256:")
    expected_size = int(asset["size"])
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "maia-local-web-chess"},
    )
    archive: Path | None = None
    digest = hashlib.sha256()
    received = 0
    print(f"Downloading {url}", flush=True)
    try:
        with tempfile.NamedTemporaryFile(
            prefix="stockfish-",
            suffix=Path(str(asset["name"])).suffix + ".part",
            dir=STOCKFISH_CACHE_DIR,
            delete=False,
        ) as temporary:
            archive = Path(temporary.name)
            with urllib.request.urlopen(request, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
    except (OSError, urllib.error.URLError) as error:
        if archive:
            archive.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download Stockfish: {error}") from error

    assert archive is not None
    if received != expected_size or digest.hexdigest() != expected_digest:
        archive.unlink(missing_ok=True)
        raise RuntimeError("The downloaded Stockfish archive failed verification")
    return archive


def _copy_stockfish_from_archive(
    archive: Path,
    asset_name: str,
    executable_name: str,
    output,
) -> None:
    if asset_name.endswith(".zip"):
        with zipfile.ZipFile(archive) as package:
            matches = [
                item
                for item in package.infolist()
                if not item.is_dir()
                and PurePosixPath(item.filename).name == executable_name
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"The Stockfish archive did not contain {executable_name}"
                )
            with package.open(matches[0]) as source:
                shutil.copyfileobj(source, output)
        return

    if asset_name.endswith((".tar", ".tar.gz")):
        with tarfile.open(archive, mode="r:*") as package:
            matches = [
                item
                for item in package.getmembers()
                if item.isfile() and PurePosixPath(item.name).name == executable_name
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"The Stockfish archive did not contain {executable_name}"
                )
            source = package.extractfile(matches[0])
            if source is None:
                raise RuntimeError(
                    f"Could not read {executable_name} from the Stockfish archive"
                )
            with source:
                shutil.copyfileobj(source, output)
        return

    raise RuntimeError(f"Unsupported Stockfish archive: {asset_name}")


def _extract_stockfish(
    archive: Path,
    asset_name: str,
    executable_name: str,
    stockfish_path: Path,
) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{executable_name}-",
            suffix=".part",
            dir=STOCKFISH_CACHE_DIR,
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            _copy_stockfish_from_archive(
                archive,
                asset_name,
                executable_name,
                output,
            )

        if os.name != "nt":
            temporary_path.chmod(temporary_path.stat().st_mode | 0o111)
        temporary_path.replace(stockfish_path)
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as error:
        raise RuntimeError(f"Could not extract Stockfish: {error}") from error
    finally:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)


def ensure_stockfish_cached() -> Path:
    asset_name, executable_name, stockfish_path = _stockfish_target()
    if stockfish_path.is_file() and stockfish_path.stat().st_size > 0:
        return stockfish_path

    STOCKFISH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag, asset = _latest_stockfish_asset(asset_name)
    archive = _download_stockfish_archive(asset)
    try:
        _extract_stockfish(archive, asset_name, executable_name, stockfish_path)
    finally:
        archive.unlink(missing_ok=True)

    metadata = {
        "release": tag,
        "asset": asset_name,
        "digest": asset["digest"],
        "source": asset["browser_download_url"],
    }
    (STOCKFISH_CACHE_DIR / "release.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return stockfish_path


def ensure_runtime_assets() -> Path:
    ensure_model_cached()
    return ensure_stockfish_cached()


def main() -> None:
    stockfish_path = ensure_runtime_assets()
    print(f"Maia cache: {MAIA_CACHE_DIR}")
    print(f"Stockfish: {stockfish_path}")


if __name__ == "__main__":
    main()
