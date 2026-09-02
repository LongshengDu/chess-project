from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path


WEB = Path(__file__).resolve().parent
MIN_NODE = 22


def node_version(executable: Path) -> tuple[int, str] | None:
    try:
        result = subprocess.run(
            [str(executable), "--version"], capture_output=True, text=True
        )
    except OSError:
        return None
    match = re.fullmatch(r"v(\d+).*", result.stdout.strip())
    return (int(match.group(1)), result.stdout.strip()) if match else None


def find_node() -> tuple[Path, str]:
    candidates = [
        Path(path)
        for path in [os.environ.get("NODE"), shutil.which("node")]
        if path
    ]
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
    raise SystemExit(
        f"Node.js {MIN_NODE}+ is required; found {', '.join(found) or 'no Node.js installation'}."
    )


def install_dependencies(node: Path, env: dict[str, str]) -> None:
    npx = shutil.which("npx", path=env["PATH"])
    locations = [
        node.parent / "node_modules" / "npm" / "bin" / "npx-cli.js",
        Path(npx).resolve().parent / "node_modules" / "npm" / "bin" / "npx-cli.js"
        if npx
        else None,
    ]
    npx_cli = next((path for path in locations if path and path.exists()), None)
    if not npx_cli:
        raise SystemExit("npm/npx is required to install the frontend dependencies.")

    package = json.loads((WEB / "package.json").read_text(encoding="utf-8"))
    pnpm = package["packageManager"]
    subprocess.run(
        [str(node), str(npx_cli), "-p", pnpm, "pnpm", "install", "--frozen-lockfile"],
        cwd=WEB,
        env={**env, "npm_config_yes": "true"},
        check=True,
    )


def main() -> None:
    node, version = find_node()
    env = {
        **os.environ,
        "PATH": str(node.parent) + os.pathsep + os.environ.get("PATH", ""),
    }
    tsc = WEB / "node_modules" / "typescript" / "bin" / "tsc"
    vite = WEB / "node_modules" / "vite" / "bin" / "vite.js"

    print(f"Using Node.js {version}", flush=True)
    if not tsc.exists() or not vite.exists():
        install_dependencies(node, env)

    subprocess.run([str(node), str(tsc), "--noEmit"], cwd=WEB, env=env, check=True)
    subprocess.run([str(node), str(vite), "build"], cwd=WEB, env=env, check=True)


if __name__ == "__main__":
    main()
