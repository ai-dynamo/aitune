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
    run(args.path, args.kind, args.test_number, args.verbose, args.dry_run, args.workflow)


def run(
    path: Path,
    kind: str,
    test_number: int,
    verbose: bool = False,
    dry_run: bool = False,
    workflow: str | None = None,
) -> None:
    """Run one zero-based entry from a functional script or example project."""
    config = _load_config(path, kind)
    try:
        entry = config.entries[test_number]
    except IndexError as exc:
        raise ValueError(f"entry {test_number} does not exist for {path}") from exc

    env = os.environ | {"AITUNE_CONSOLE_OUTPUT": "1"} | config.environment
    _validate_requested_workflow(path, kind, config, workflow)
    _install_dependencies(path, kind, config, verbose, dry_run, workflow)
    _save_requirements(verbose, dry_run)

    run_kwargs: dict[str, Any] = {"cwd": path if kind == "project" else None, "env": env}
    if kind == "project":
        _run_project(path, entry, workflow, verbose, dry_run, run_kwargs)
    else:
        _run_command(_command(path, kind, entry), verbose, dry_run, **run_kwargs)


def _validate_requested_workflow(
    path: Path,
    kind: str,
    config: FunctionalTestConfig,
    workflow: str | None,
) -> None:
    if kind != "project":
        return
    configured_workflows = [configured_workflow.name for configured_workflow in config.workflows]
    if workflow is None and config.workflows:
        raise ValueError(f"Project {path} requires one of its configured workflows: {', '.join(configured_workflows)}")
    if workflow is not None and workflow not in configured_workflows:
        raise ValueError(f"Workflow {workflow} is not configured for project {path}")


def _run_project(
    path: Path,
    entry: FunctionalVariantConfig,
    workflow: str | None,
    verbose: bool,
    dry_run: bool,
    run_kwargs: dict[str, Any],
) -> None:
    if workflow is None:
        for script in ("tune", "inference"):
            _run_command(_command(path, "project", entry, script), verbose, dry_run, **run_kwargs)
        if (path / "run_dynamo.sh").is_file():
            _run_command(["./run_dynamo.sh"], verbose, dry_run, **run_kwargs)
        return

    target = "python" if workflow == "dynamo" else "triton"
    _run_command(_command(path, "project", entry, "tune", {"target": target}), verbose, dry_run, **run_kwargs)
    if workflow == "dynamo":
        _run_command(["./run_dynamo.sh"], verbose, dry_run, **run_kwargs)
        return

    model_repository = _model_repository_path(path)
    if not dry_run:
        model_repository.parent.mkdir(parents=True, exist_ok=True)
    # Publishing reads the package produced by tuning and must run once, even for a distributed tuning variant.
    _run_command(
        _command(
            path,
            "project",
            entry,
            "triton-model-store",
            {"model-repository": model_repository},
            include_entry_arguments=False,
            use_launcher=False,
        ),
        verbose,
        dry_run,
        **run_kwargs,
    )
    _run_triton_validation(path, entry, verbose, dry_run, run_kwargs)


def _run_triton_validation(
    path: Path,
    entry: FunctionalVariantConfig,
    verbose: bool,
    dry_run: bool,
    run_kwargs: dict[str, Any],
) -> None:
    model_repository = _model_repository_path(path)
    triton_run_kwargs = dict(run_kwargs)
    triton_run_kwargs["env"] = run_kwargs["env"] | {"MODEL_REPOSITORY": str(model_repository)}
    _run_command(["./run_triton.sh", *_arguments(entry.arguments)], verbose, dry_run, **triton_run_kwargs)


def _model_repository_path(path: Path) -> Path:
    artifacts_dir = Path(os.environ.get("AITUNE_ARTIFACTS_DIR", "artifacts")).resolve()
    return artifacts_dir / path.name / "model_repository"


def _install_dependencies(
    path: Path,
    kind: str,
    config: FunctionalTestConfig,
    verbose: bool,
    dry_run: bool,
    workflow: str | None = None,
) -> None:
    if kind != "project" or workflow != "triton":
        _run_command([sys.executable, "-m", "pip", "install", "--group", "functional-test"], verbose, dry_run)

    if kind == "project":
        _run_command([sys.executable, "-m", "pip", "install", "examples/common"], verbose, dry_run)
        extra = workflow or "dynamo"
        _run_command([sys.executable, "-m", "pip", "install", f"{path}[{extra}]"], verbose, dry_run)

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


def _command(
    path: Path,
    kind: str,
    entry: FunctionalVariantConfig,
    script: str | None = None,
    extra_arguments: dict[str, Any] | None = None,
    include_entry_arguments: bool = True,
    use_launcher: bool = True,
) -> list[str]:
    command_arguments = dict(entry.arguments) if include_entry_arguments else {}
    command_arguments.update(extra_arguments or {})
    arguments = _arguments(command_arguments)
    if kind == "script":
        return [sys.executable, str(path), *arguments]

    if script is None:
        raise ValueError("Project commands require a script name")
    module = _project_module(path, script)
    if entry.launcher and use_launcher:
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
    parser.add_argument("--workflow", choices=("legacy", "dynamo", "triton"), default="legacy")
    parser.add_argument("--is-custom-docker-image", type=json.loads, default=False)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workflow == "legacy":
        args.workflow = None
    return args


if __name__ == "__main__":
    main()
