"""Build the frontend, preparing its runtime and dependencies when necessary."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web.settings import FRONTEND
from web.build_node import NodeRuntime
from web.build_packages import FrontendPackages


def main() -> None:
    upstream = FRONTEND.parent / "deps" / "maia-platform-frontend" / "src"
    if not upstream.is_dir():
        raise SystemExit("Run: git submodule update --init --depth 1 deps/maia-platform-frontend")
    runtime = NodeRuntime()
    packages = FrontendPackages()
    found_node = runtime.find_node()
    if found_node:
        node, version = found_node
    else:
        print(f"Node.js {runtime.minimum_version}+ was not found; installing it automatically.", flush=True)
        node, version = runtime.install_node()
    env = {
        **os.environ,
        "PATH": str(node.parent) + os.pathsep + os.environ.get("PATH", ""),
    }
    package = packages.package_manifest()
    missing = packages.missing_dependencies(package)
    tsc = FRONTEND / "node_modules" / "typescript" / "bin" / "tsc"
    vite = FRONTEND / "node_modules" / "vite" / "bin" / "vite.js"

    print(f"Using Node.js {version}", flush=True)
    if missing:
        if not packages.has_package_runner(node, env):
            node, version = runtime.install_node()
            env["PATH"] = str(node.parent) + os.pathsep + os.environ.get("PATH", "")
            print(f"Using Node.js {version} with its bundled package manager", flush=True)
        print(f"Missing or outdated packages: {', '.join(missing)}", flush=True)
        packages.install_dependencies(node, env, package)

    subprocess.run([str(node), str(tsc), "--noEmit"], cwd=FRONTEND, env=env, check=True)
    subprocess.run([str(node), str(vite), "build"], cwd=FRONTEND, env=env, check=True)


if __name__ == "__main__":
    main()
