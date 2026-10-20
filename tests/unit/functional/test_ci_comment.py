# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the functional CI PR comment formatter."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("xml_attrs", "expected"),
    [
        (
            'tests="7" failures="2" errors="1" skipped="1"',
            ":x: Functional tests: 3 passed, 1 skipped, 3 failed",
        ),
        (
            'tests="42" failures="0" errors="0" skipped="0"',
            ":white_check_mark: Functional tests: 42 passed, 0 failed",
        ),
        ('tests="0" skipped="0"', ":x: Functional tests: 0 passed, 0 failed"),
    ],
)
def test_comment_counts_failures_and_errors_and_links_report(tmp_path, xml_attrs, expected):
    report = tmp_path / "report.xml"
    report.write_text(
        f"<testsuites><testsuite {xml_attrs} /></testsuites>",
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[3] / ".github/scripts/functional_ci_comment.py"
    url = "https://github.com/org/repo/actions/runs/123/artifacts/456"
    result = subprocess.run(
        [sys.executable, str(script), str(report), url],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == f"{expected}\n\n[Feature matrix report]({url})"
