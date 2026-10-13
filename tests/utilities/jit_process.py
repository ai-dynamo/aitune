# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run JIT scenarios in fresh interpreters instead of resetting live modules."""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from _pytest.reports import TestReport

_REPORT_PATH = "AITUNE_JIT_TEST_REPORTS"


def pytest_runtest_logreport(report):
    """Forward ordinary pytest outcomes, including setup failures and skips."""
    if report_path := os.environ.get(_REPORT_PATH):
        with Path(report_path).open("a") as stream:
            stream.write(json.dumps(report._to_json()) + "\n")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    """Spawn each stateful JIT item, including its normal fixtures and doctests."""
    if os.environ.get(_REPORT_PATH):
        return None
    if item.path.suffix != ".py":
        return None
    source = item.path.read_text()
    names = ("prepare_for_jit_tuning", "register_for_jit_tuning", "Patcher", "PatchedModule", "InspectModule")
    if not any(name in source for name in names):
        return None
    item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    with TemporaryDirectory(prefix="aitune-jit-test-") as directory:
        report_path = Path(directory) / "reports.jsonl"
        env = {**os.environ, _REPORT_PATH: str(report_path), "PYTEST_ADDOPTS": ""}
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "--import-mode=importlib",
            "--doctest-modules",
            item.nodeid,
        ]
        try:
            result = subprocess.run(
                command, cwd=item.config.rootpath, env=env, capture_output=True, text=True, timeout=600
            )
            output, failed = result.stdout + result.stderr, result.returncode != 0
        except subprocess.TimeoutExpired as error:
            output = "JIT test worker exceeded 600 seconds.\n" + (error.stdout or b"").decode(errors="replace")
            output += (error.stderr or b"").decode(errors="replace")
            failed = True
        reports = []
        if report_path.exists():
            for line in report_path.read_text().splitlines():
                reports.append(TestReport._from_json(json.loads(line)))
        if not reports or (failed and not any(report.failed for report in reports)):
            reports.append(TestReport(item.nodeid, item.location, {}, "failed", output, "call"))
        for report in reports:
            item.ihook.pytest_runtest_logreport(report=report)
    item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True
