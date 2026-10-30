# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the toy-model functional suite.

The root `conftest.py` already isolates the AITune cache, module registry, and JIT config per test.
"""

import datetime
import json
import os
from pathlib import Path

import pytest
import torch

from aitune.torch.config import config
from aitune.torch.module_registry import MODULE_REGISTRY
from tests.functional_ci.backends import BACKENDS

_RANK = {"passed": 0, "skipped": 1, "failed": 2}
_cells: dict[str, dict] = {}
_item_cells: dict[str, list[str]] = {}


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Return the first GPU; the suite fails instead of skipping without CUDA."""
    if not torch.cuda.is_available():
        pytest.fail("tests/functional_ci requires CUDA", pytrace=False)
    return torch.device("cuda:0")


@pytest.fixture(autouse=True)
def isolated_tuning_report(monkeypatch, tmp_path):
    """Keep tuning reports out of the user cache and release backend resources after each case."""
    monkeypatch.setattr(config, "_tuning_data_output_path", tmp_path / "tuning-report.json")
    yield
    for module in MODULE_REGISTRY.modules.values():
        module.deactivate()


@pytest.fixture(scope="session", autouse=True)
def fp32_convolutions():
    """Disable TF32 in cuDNN convolutions for the whole session."""
    # On Torch 2.11 and 2.14, Inductor switches convolutions to channels_last, where cuDNN picks TF32 kernels; for the
    # eager NCHW convolutions it picks fp32 kernels, although `allow_tf32` defaults to True. The outputs differ by
    # about 5e-4 and fail the 1e-5 tolerance of the inductor cases.
    # Set once and never reset: on Torch 2.8, Inductor imports `torch.testing._internal` while compiling, which blocks
    # later changes to the cuDNN flags.
    torch.backends.cudnn.allow_tf32 = False


def pytest_configure(config):
    config.addinivalue_line("markers", "matrix(*cells): feature-matrix cells that this test proves")


def _cell_ids(item) -> list[str]:
    marker = item.get_closest_marker("matrix")
    if marker is None:
        return []

    callspec = getattr(item, "callspec", None)
    params = callspec.params if callspec is not None else {}
    case = params.get("case")
    backend = next((name for name, known in BACKENDS.items() if known is case), None)
    ids = []
    for template in marker.args:
        if "/" not in template:
            template = f"{template}/{{backend}}"
        if backend is None and "{backend" in template:
            raise pytest.UsageError(f"{item.nodeid}: matrix row {template!r} needs a resolvable backend")
        try:
            ids.append(template.format(backend=backend, **params))
        except KeyError as exc:
            raise pytest.UsageError(
                f"{item.nodeid}: cannot resolve matrix row {template!r}: missing {exc.args[0]!r}"
            ) from exc
    return ids


def pytest_collection_modifyitems(config, items):
    _item_cells.clear()
    for item in items:
        if item.get_closest_marker("matrix") is not None:
            _item_cells[item.nodeid] = _cell_ids(item)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if report.when == "call" or report.outcome != "passed":
        for cell_id in _item_cells.get(item.nodeid, []):
            cell = _cells.setdefault(cell_id, {"state": "passed", "tests": set()})
            cell["tests"].add(item.nodeid)
            if _RANK[report.outcome] > _RANK[cell["state"]]:
                cell["state"], cell["culprit"] = report.outcome, item.nodeid.split("/")[-1]
                cell["reason"] = _reason(report, call)
    return report


def _reason(report, call) -> str:
    if hasattr(report, "wasxfail"):
        return report.wasxfail
    if report.skipped and isinstance(report.longrepr, tuple):
        return report.longrepr[2].removeprefix("Skipped: ")
    if call.excinfo is not None:
        return call.excinfo.exconly().splitlines()[0]
    return str(report.longrepr)


def pytest_sessionfinish(session):
    path = os.environ.get("AITUNE_MATRIX_RESULTS")
    if not path or not _cells:
        return
    run_url = os.environ.get("GITHUB_RUN_ID") and (
        "{GITHUB_SERVER_URL}/{GITHUB_REPOSITORY}/actions/runs/{GITHUB_RUN_ID}".format(**os.environ)
    )
    cells = {
        cell_id: {
            "state": cell["state"],
            "run": (
                f"{cell['state']}: {cell['culprit']} ({cell['reason']})"
                if "culprit" in cell
                else f"passed: {len(cell['tests'])} test(s)"
            ),
        }
        for cell_id, cell in sorted(_cells.items())
    }
    run = {
        "date": datetime.date.today().isoformat(),
        "source": run_url or os.environ.get("GITHUB_SHA", "local pytest run"),
    }
    Path(path).write_text(json.dumps({"run": run, "cells": cells}, indent=1), encoding="utf-8")
