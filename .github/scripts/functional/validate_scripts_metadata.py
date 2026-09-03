# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate PEP-723 / ``[tool.aitune]`` metadata for functional tests and examples."""

# /// script
# dependencies = ["pydantic"]
# ///

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_script_metadata import FunctionalTestConfig  # noqa: E402
from pydantic import ValidationError  # noqa: E402

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # pytype: disable=import-error

logger = logging.getLogger("validate_scripts_metadata")

DEFAULT_SCRIPTS_ROOT = Path("tests/functional")
DEFAULT_PROJECTS_ROOT = Path("examples")
SCRIPT_GLOB = "[0-9][0-9][0-9]_*.py"


def collect_scripts(roots: list[Path]) -> list[Path]:
    """Collect numbered functional-test scripts under each root."""
    return sorted(script for root in roots for script in root.rglob(SCRIPT_GLOB))


def collect_projects(roots: list[Path]) -> list[Path]:
    """Collect example ``pyproject.toml`` files under each root."""
    return sorted(project for root in roots for project in root.glob("*/pyproject.toml"))


def _validate_one(path: Path) -> str | None:
    try:
        if path.name == "pyproject.toml":
            FunctionalTestConfig.from_project(path)
        else:
            FunctionalTestConfig.from_script(path)
    except (ValidationError, ValueError, tomllib.TOMLDecodeError) as exc:
        return f"{path}: {exc}"
    return None


def validate(scripts: list[Path], projects: list[Path]) -> int:
    """Validate functional-test scripts and example projects. Returns error count."""
    failures = [error for path in [*scripts, *projects] if (error := _validate_one(path))]
    for error in failures:
        print(error, file=sys.stderr)  # noqa: T201
    print(f"validated {len(scripts)} scripts, {len(projects)} projects, {len(failures)} errors")  # noqa: T201
    return len(failures)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--scripts-path",
        action="append",
        type=Path,
        default=[],
        help="Functional script directory. Repeatable. Default: tests/functional",
    )
    parser.add_argument(
        "-p",
        "--projects-path",
        action="append",
        type=Path,
        default=[],
        help="Examples directory. Repeatable. Default: examples",
    )
    return parser.parse_args()


def main() -> None:
    """Validate the functional-test and example metadata corpus."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    scripts = collect_scripts(args.scripts_path or [DEFAULT_SCRIPTS_ROOT])
    projects = collect_projects(args.projects_path or [DEFAULT_PROJECTS_ROOT])
    if not scripts and not projects:
        raise SystemExit("no scripts or projects found")
    raise SystemExit(1 if validate(scripts, projects) else 0)


if __name__ == "__main__":
    main()
