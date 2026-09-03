# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate GitHub Actions matrix entries for functional tests and examples.

Reads PEP-723 script metadata and ``[tool.aitune]`` project metadata, then emits
JSON suitable for ``strategy.matrix.include`` or ``$GITHUB_OUTPUT``.
"""

# /// script
# dependencies = ["pydantic"]
# ///

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_script_metadata import (
    DEFAULT_DOCKER_IMAGE,
    FunctionalTestConfig,
    Scope,
    get_scope,
)

logger = logging.getLogger("generate_github_actions_matrix")

DEFAULT_RUNNER = "prod-aitune-tester-rtx-pro-4500-v1"
DEFAULT_TIMEOUT_MINUTES = 40


def _timeout_to_minutes(timeout: str | None) -> int:
    if not timeout:
        return DEFAULT_TIMEOUT_MINUTES
    total_minutes = 0
    for amount, unit in re.findall(r"(\d+)\s*([hms])", timeout.lower()):
        value = int(amount)
        if unit == "h":
            total_minutes += value * 60
        elif unit == "m":
            total_minutes += value
        elif unit == "s":
            total_minutes += max(1, value // 60)
    return total_minutes or DEFAULT_TIMEOUT_MINUTES


def _environment_for_job(config: FunctionalTestConfig) -> dict[str, str]:
    variables = dict(config.environment)
    if "AITUNE_CONSOLE_OUTPUT" not in variables:
        variables["AITUNE_CONSOLE_OUTPUT"] = "1"
    return variables


def _matrix_entry(
    *,
    entry_id: str,
    test_number: int,
    kind: str,
    path: str,
    config: FunctionalTestConfig,
    requested_scope: Scope,
) -> dict[str, Any]:
    should_run = config.scope.in_scope(requested_scope) if config.scope is not None else True
    docker_image = config.docker_image or DEFAULT_DOCKER_IMAGE
    runner = config.runner or DEFAULT_RUNNER

    entry: dict[str, Any] = {
        "id": entry_id,
        "test_number": test_number,
        "docker_image": docker_image,
        "runner": runner,
        "environment": json.dumps(_environment_for_job(config)),
        "kind": kind,
        "path": path,
        "allow_failure": config.allow_failure or not should_run,
        "timeout_minutes": _timeout_to_minutes(config.timeout),
        "use_gated_hf_token": config.use_gated_hf_token,
    }
    return entry


def _make_script_entries(
    namespace: str,
    script: Path,
    config: FunctionalTestConfig,
    requested_scope: Scope,
) -> list[dict[str, Any]]:
    if config.skip:
        return []

    jobs: list[dict[str, Any]] = []
    for index, _ in enumerate(config.entries, start=1):
        entry_id = f"{namespace}_{script.stem}_{index:03d}"
        jobs.append(
            _matrix_entry(
                entry_id=entry_id,
                test_number=index - 1,  # zero-based index for GitHub Actions matrix
                kind="script",
                path=script.as_posix(),
                config=config,
                requested_scope=requested_scope,
            )
        )
    return jobs


def _make_project_entries(
    namespace: str,
    project: Path,
    config: FunctionalTestConfig,
    requested_scope: Scope,
) -> list[dict[str, Any]]:
    if config.skip:
        return []

    parent_dir = project.parent
    name = parent_dir.name

    jobs: list[dict[str, Any]] = []
    for index, _ in enumerate(config.entries, start=1):
        entry_id = f"{namespace}_{name}_inference_{index:03d}"
        jobs.append(
            _matrix_entry(
                entry_id=entry_id,
                test_number=index - 1,  # zero-based index for GitHub Actions matrix
                kind="project",
                path=parent_dir.as_posix(),
                config=config,
                requested_scope=requested_scope,
            )
        )
    return jobs


def get_scripts(scripts_paths: list[str]):
    """Yield all script paths in functional script directories."""
    for script_path in scripts_paths:
        path = Path(script_path)
        if not path.is_dir():
            logger.error("Invalid scripts path: %s", script_path)
            continue
        namespace = path.stem
        for script in sorted(path.glob("[0-9][0-9][0-9]_*.py")):
            yield namespace, script


def get_projects(projects_paths: list[str]):
    """Yield all project paths (pyproject.toml) in example directories."""
    for project_path in projects_paths:
        path = Path(project_path)
        namespace = path.stem
        for pyproject in sorted(path.glob("*/pyproject.toml")):
            yield namespace, pyproject


def generate_matrix(
    *,
    script_paths: list[str],
    projects_paths: list[str],
    default_docker_image: str,
    default_scope: str,
    example_default_scope: str,
    test_scope_env: str,
    example_scope_env: str,
) -> list[dict[str, Any]]:
    """Generate full GitHub Actions job matrix for tests and examples."""
    requested_test_scope = get_scope({"scope": os.environ.get(test_scope_env, default_scope)})
    requested_example_scope = get_scope({"scope": os.environ.get(example_scope_env, example_default_scope)})
    logger.info("Requested test scope: %s", requested_test_scope)
    logger.info("Requested example scope: %s", requested_example_scope)

    matrix: list[dict[str, Any]] = []
    for namespace, script in get_scripts(script_paths):
        config = FunctionalTestConfig.from_script(script, default_scope, default_docker_image)
        matrix.extend(_make_script_entries(namespace, script, config, requested_test_scope))

    for namespace, project in get_projects(projects_paths):
        config = FunctionalTestConfig.from_project(project, example_default_scope, default_docker_image)
        matrix.extend(_make_project_entries(namespace, project, config, requested_example_scope))

    return matrix


def write_github_output(path: str | Path, matrix: list[dict[str, Any]]) -> None:
    """Write GitHub-formatted outputs to file for matrix and count."""
    output_path = Path(path)
    payload = json.dumps(matrix, separators=(",", ":"))
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(f"matrix={payload}\n")
        handle.write(f"matrix_count={len(matrix)}\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-json",
        type=str,
        help="Write the matrix JSON to a file.",
    )
    parser.add_argument(
        "-s",
        "--scripts-path",
        action="append",
        default=[],
        help="Functional script directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "-p",
        "--projects-path",
        action="append",
        default=[],
        help="Examples/projects directory. Can be passed multiple times.",
    )
    parser.add_argument("--default-docker-image", default=DEFAULT_DOCKER_IMAGE)
    parser.add_argument("--default-scope", default="L0")
    parser.add_argument("--example-default-scope", default="L1")
    parser.add_argument("--test-scope-env", default="AITUNE_TEST_SCOPE")
    parser.add_argument("--example-scope-env", default="AITUNE_EXAMPLE_SCOPE")
    parser.add_argument("--github-output", help="Path to the GitHub Actions output file.")
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the generated matrix JSON to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entrypoint for matrix generation script."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    matrix = generate_matrix(
        script_paths=args.scripts_path,
        projects_paths=args.projects_path,
        default_docker_image=args.default_docker_image,
        default_scope=args.default_scope,
        example_default_scope=args.example_default_scope,
        test_scope_env=args.test_scope_env,
        example_scope_env=args.example_scope_env,
    )
    logger.info("Generated %d matrix jobs", len(matrix))
    if len(matrix) > 256:
        raise ValueError(f"Generated matrix has {len(matrix)} jobs; GitHub Actions limit is 256.")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(matrix, indent=2), encoding="utf-8")

    if args.github_output:
        write_github_output(args.github_output, matrix)

    if args.stdout:
        print(json.dumps(matrix, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
