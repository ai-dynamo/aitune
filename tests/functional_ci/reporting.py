# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect environment metadata and retain functional case properties in JUnit reports."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.functional_ci.capabilities import (
    FunctionalProfile,
    _gpu_environment,
    capability_reason,
    check_environment,
    get_functional_profile,
    profile_data,
)

_CASE_PROPERTIES: pytest.StashKey[list[tuple[str, str]]] = pytest.StashKey()
_AREA_MARKERS = {
    "backend_lifecycle",
    "shapes",
    "module_topology",
    "jit",
    "strategy_fallback",
    "checkpoints",
    "artifacts",
    "precision",
    "cuda_graphs",
    "attention",
    "distributed",
    "cross_feature",
}


def case_properties(item: pytest.Item) -> list[tuple[str, str]]:
    """Convert functional markers to stable JUnit key/value properties."""
    props: dict[str, str] = {"case_id": item.name}
    areas = sorted(marker.name for marker in item.iter_markers() if marker.name in _AREA_MARKERS)
    if areas:
        props["areas"] = ",".join(areas)
    case_marker = next(item.iter_markers("functional_case"), None)
    if case_marker:
        props.update({str(key): str(value) for key, value in case_marker.kwargs.items()})
        backend = props.get("backend", "")
        if "," in backend:
            props["candidate_backends"] = backend
    cap_marker = next(item.iter_markers("requires_capability"), None)
    if cap_marker:
        capability = str(cap_marker.kwargs.get("name", cap_marker.args[0] if cap_marker.args else ""))
        props["required_capability"] = capability
    return sorted(props.items())


def write_run_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Write run metadata as UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_run_metadata(profile: FunctionalProfile) -> dict[str, Any]:
    """Gather environment metadata without importing AITune."""
    environment_errors: list[str] = []
    try:
        environment = check_environment(profile)
    except Exception as error:
        environment_errors.append(str(error))
        gpus, driver, cuda = _gpu_environment()
        environment = {"gpus": gpus, "gpu_count": len(gpus), "driver_version": driver, "cuda_version": cuda}
    if environment["driver_version"] is None:
        environment_errors.append("GPU driver version unavailable")
    if environment["cuda_version"] is None:
        environment_errors.append("CUDA version unavailable")

    distributions = {
        "torch": "torch",
        "torch_tensorrt": "torch-tensorrt",
        "tensorrt": "tensorrt",
        "onnx": "onnx",
        "onnxruntime": "onnxruntime-gpu",
        "torchao": "torchao",
        "modelopt": "nvidia-modelopt",
        "polygraphy": "polygraphy",
    }
    versions: dict[str, str | None] = {}
    for module, distribution in distributions.items():
        try:
            versions[module] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[module] = None
            environment_errors.append(f"Installed version unavailable for {module} ({distribution})")
    try:
        commit_sha = (
            os.environ.get("GITHUB_SHA")
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        )
    except Exception:
        commit_sha = None
        environment_errors.append("Commit SHA unavailable")
    return {
        "schema_version": 1,
        "commit_sha": commit_sha,
        "ci_run_id": os.environ.get("GITHUB_RUN_ID"),
        "ci_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "container_ref": os.environ.get("AITUNE_FUNCTIONAL_CONTAINER_REF"),
        "container_digest": os.environ.get("AITUNE_FUNCTIONAL_CONTAINER_DIGEST"),
        **environment,
        "library_versions": versions,
        "capability_profile": profile_data(profile),
        "environment_errors": environment_errors,
    }


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--functional-run-metadata", default="functional-ci.run.json")


def pytest_sessionstart(session: pytest.Session) -> None:
    profile = get_functional_profile()
    metadata = collect_run_metadata(profile)
    write_run_metadata(Path(session.config.getoption("--functional-run-metadata")), metadata)
    try:
        check_environment(profile)
    except Exception as error:
        pytest.exit(str(error), returncode=4)


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:
    names = [item.name for item in items]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise pytest.UsageError(f"duplicate case ID(s): {', '.join(duplicates)}")
    profile = get_functional_profile()
    for item in items:
        props = case_properties(item)
        item.stash[_CASE_PROPERTIES] = props
        cap_marker = next(item.iter_markers("requires_capability"), None)
        if cap_marker:
            capability = props[[key for key, _ in props].index("required_capability")][1]
            reason = capability_reason(profile, capability)
            if reason:
                item.add_marker(pytest.mark.skip(reason=f"{item.name}: {reason}"))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]):
    outcome = yield
    report = outcome.get_result()
    properties = item.stash.get(_CASE_PROPERTIES, case_properties(item))
    existing_properties = {name for name, _ in report.user_properties}
    report.user_properties.extend((name, value) for name, value in properties if name not in existing_properties)
    if report.failed:
        cache_dir = getattr(item, "_functional_cache_dir", None)
        if cache_dir and Path(cache_dir).is_dir():
            target = Path("functional-ci-logs") / item.name
            target.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copytree(cache_dir, target / "cache", dirs_exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = collect_run_metadata(get_functional_profile())
    write_run_metadata(args.output, metadata)


if __name__ == "__main__":
    main()
