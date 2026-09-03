# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one functional-test matrix entry."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metadata import FunctionalTestConfig, FunctionalVariantConfig  # noqa: E402

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # pytype: disable=import-error


def _arguments(arguments: dict[str, Any]) -> list[str]:
    return [f"--{name}={json.dumps(value, separators=(',', ':'))}" for name, value in arguments.items()]


def _load_config(path: Path, kind: str) -> FunctionalTestConfig:
    if kind == "script":
        return FunctionalTestConfig.from_script(path)
    return FunctionalTestConfig.from_project(path / "pyproject.toml")


def _project_module(path: Path) -> str:
    project = tomllib.loads((path / "pyproject.toml").read_text(encoding="utf-8"))
    target = project["project"]["scripts"]["inference"]
    return target.partition(":")[0]


def _command(path: Path, kind: str, entry: FunctionalVariantConfig) -> list[str]:
    arguments = _arguments(entry.arguments)
    if kind == "script":
        return [sys.executable, str(path), *arguments]
    module = _project_module(path)
    if entry.launcher:
        command = [sys.executable, "-m", entry.launcher]
        if entry.processes is not None:
            command.extend(["--standalone", f"--nproc-per-node={entry.processes}"])
        return [*command, "--module", module, *arguments]
    return [sys.executable, "-m", module, *arguments]


def _run_command(command: list[str], verbose: bool, dry_run: bool, **kwargs: Any) -> None:
    if verbose:
        print(f"+ {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, check=True, **kwargs)


def _install_dist(verbose: bool = False, dry_run: bool = False) -> None:
    wheels = sorted(Path("dist").glob("*.whl"))
    if not wheels:
        raise FileNotFoundError("no wheel found in dist/")
    _run_command(
        [sys.executable, "-m", "pip", "install", *(str(wheel) for wheel in wheels)],
        verbose,
        dry_run,
    )


def _install_dependencies(
    path: Path,
    kind: str,
    config: FunctionalTestConfig,
    is_custom_docker_image: bool,
    verbose: bool,
    dry_run: bool,
) -> None:
    if is_custom_docker_image:
        _install_dist(verbose, dry_run)
    if kind == "project":
        _run_command(
            [sys.executable, "-m", "pip", "install", "--editable", str(path)],
            verbose,
            dry_run,
        )
        return
    if config.dependencies:
        _run_command(
            [sys.executable, "-m", "pip", "install", *config.dependencies],
            verbose,
            dry_run,
        )
    for install in config.pip_install:
        _run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                *install.get("flags", []),
                *install.get("packages", []),
            ],
            verbose,
            dry_run,
        )


def run(
    path: Path,
    kind: str,
    test_number: int,
    is_custom_docker_image: bool = False,
    verbose: bool = False,
    dry_run: bool = False,
) -> None:
    """Run one zero-based entry from a functional script or example project."""
    config = _load_config(path, kind)
    try:
        entry = config.entries[test_number]
    except IndexError as exc:
        raise ValueError(f"entry {test_number} does not exist for {path}") from exc

    env = os.environ | {"AITUNE_CONSOLE_OUTPUT": "1"} | config.environment
    _install_dependencies(path, kind, config, is_custom_docker_image, verbose, dry_run)
    _run_command(
        _command(path, kind, entry),
        verbose,
        dry_run,
        cwd=path if kind == "project" else None,
        env=env,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--kind", choices=("script", "project"), required=True)
    parser.add_argument("--test-number", type=int, required=True)
    parser.add_argument("--is-custom-docker-image", type=json.loads, default=False)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the selected functional matrix entry."""
    args = parse_args()
    run(args.path, args.kind, args.test_number, args.is_custom_docker_image, args.verbose, args.dry_run)


if __name__ == "__main__":
    main()
