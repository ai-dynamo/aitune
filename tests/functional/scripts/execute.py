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


def main() -> None:
    """Run the selected functional matrix entry."""
    args = parse_args()
    run(args.path, args.kind, args.test_number, args.verbose, args.dry_run)


def run(path: Path, kind: str, test_number: int, verbose: bool = False, dry_run: bool = False) -> None:
    """Run one zero-based entry from a functional script or example project."""
    config = _load_config(path, kind)
    try:
        entry = config.entries[test_number]
    except IndexError as exc:
        raise ValueError(f"entry {test_number} does not exist for {path}") from exc

    env = os.environ | {"AITUNE_CONSOLE_OUTPUT": "1"} | config.environment
    _install_dependencies(path, kind, config, verbose, dry_run)
    _save_requirements(verbose, dry_run)

    run_kwargs: dict[str, Any] = {"cwd": path if kind == "project" else None, "env": env}
    if kind == "project":
        for script in ("tune", "inference"):
            _run_command(_command(path, kind, entry, script), verbose, dry_run, **run_kwargs)
        if (path / "run_dynamo.sh").is_file():
            _run_command(["./run_dynamo.sh"], verbose, dry_run, **run_kwargs)
    else:
        _run_command(_command(path, kind, entry), verbose, dry_run, **run_kwargs)


def _install_dependencies(path: Path, kind: str, config: FunctionalTestConfig, verbose: bool, dry_run: bool) -> None:

    _run_command([sys.executable, "-m", "pip", "install", "--group", "functional-test"], verbose, dry_run)

    if kind == "project":
        _run_command([sys.executable, "-m", "pip", "install", "examples/common"], verbose, dry_run)
        _run_command([sys.executable, "-m", "pip", "install", str(path) + "[dynamo]"], verbose, dry_run)

    if config.dependencies:
        _run_command([sys.executable, "-m", "pip", "install", *config.dependencies], verbose, dry_run)

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


def _save_requirements(verbose: bool = False, dry_run: bool = False) -> None:
    _run_command(
        [sys.executable, "-m", "pip", "freeze", "--all", "--no-input", "--local", "--quiet"],
        verbose,
        dry_run,
        stdout_path=Path("functional_test_requirements.txt"),
    )


def _command(path: Path, kind: str, entry: FunctionalVariantConfig, script: str | None = None) -> list[str]:
    arguments = _arguments(entry.arguments)
    if kind == "script":
        return [sys.executable, str(path), *arguments]

    module = _project_module(path, script)
    if entry.launcher:
        command = [sys.executable, "-m", "torch.distributed.run"]
        if entry.processes is not None:
            command.extend(["--standalone", f"--nproc-per-node={entry.processes}"])
        return [*command, "--module", module, *arguments]
    return [sys.executable, "-m", module, *arguments]


def _arguments(arguments: dict[str, Any]) -> list[str]:
    rendered = []
    for key, value in arguments.items():
        if value is True:
            rendered.append(f"--{key}")
        elif value is not False and value is not None:
            rendered.append(f"--{key}={value}")
    return rendered


def _project_module(path: Path, script: str) -> str:
    project = tomllib.loads((path / "pyproject.toml").read_text(encoding="utf-8"))
    target = project["project"]["scripts"][script]
    return target.partition(":")[0]


def _run_command(
    command: list[str], verbose: bool, dry_run: bool, stdout_path: Path | None = None, **kwargs: Any
) -> None:
    if verbose:
        redirect = f" > {stdout_path}" if stdout_path else ""
        print(f"+ {shlex.join(command)}{redirect} @ {kwargs.get('cwd', '.')}", flush=True)
    if dry_run:
        return
    if stdout_path is None:
        subprocess.run(command, check=True, **kwargs)
        return
    with stdout_path.open("w", encoding="utf-8") as handle:
        subprocess.run(command, check=True, stdout=handle, **kwargs)


def _load_config(path: Path, kind: str) -> FunctionalTestConfig:
    if kind == "script":
        return FunctionalTestConfig.from_script(path)
    return FunctionalTestConfig.from_project(path / "pyproject.toml")


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


if __name__ == "__main__":
    main()
