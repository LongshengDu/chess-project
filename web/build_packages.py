"""Pinned frontend dependency discovery and package-manager installation."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from .settings import FRONTEND


class FrontendPackages:
    def __init__(self, directory=FRONTEND):
        self.directory = directory

    @staticmethod
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

    @staticmethod
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

    def has_package_runner(self, node: Path, env: dict[str, str]) -> bool:
        return bool(self.find_corepack_cli(node) or self.find_npx_cli(node, env))

    def package_manifest(self) -> dict[str, object]:
        return json.loads((self.directory / "package.json").read_text(encoding="utf-8"))

    def missing_dependencies(self, package: dict[str, object]) -> list[str]:
        required = {
            **package.get("dependencies", {}),
            **package.get("devDependencies", {}),
        }
        missing = []
        for name, wanted_version in required.items():
            installed_manifest = self.directory / "node_modules" / Path(*name.split("/")) / "package.json"
            try:
                installed_version = json.loads(installed_manifest.read_text(encoding="utf-8"))["version"]
            except (FileNotFoundError, KeyError, json.JSONDecodeError):
                missing.append(name)
                continue
            if re.fullmatch(r"\d+\.\d+\.\d+(?:[-+].+)?", str(wanted_version)):
                if installed_version != wanted_version:
                    missing.append(name)
        return missing

    def install_dependencies(self,
        node: Path, env: dict[str, str], package: dict[str, object]
    ) -> None:
        package_manager = str(package["packageManager"])
        manager = package_manager.partition("@")[0]
        corepack_cli = self.find_corepack_cli(node)
        if corepack_cli:
            command = [str(node), str(corepack_cli), manager, "install", "--frozen-lockfile"]
        else:
            npx_cli = self.find_npx_cli(node, env)
            if not npx_cli:
                raise SystemExit("The Node.js installation does not include Corepack or npm/npx.")
            # npm 6's npx misparses --yes; npm_config_yes below works with both runners.
            command = [
                str(node),
                str(npx_cli),
                "--package",
                package_manager,
                manager,
                "install",
                "--frozen-lockfile",
            ]

        print(f"Installing frontend dependencies with {package_manager}", flush=True)
        subprocess.run(
            command,
            cwd=self.directory,
            env={
                **env,
                "CI": env.get("CI") or "true",
                "COREPACK_ENABLE_DOWNLOAD_PROMPT": "0",
                "npm_config_yes": "true",
            },
            check=True,
        )
