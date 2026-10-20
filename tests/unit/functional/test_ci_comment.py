# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the functional CI PR comment formatter."""

import subprocess
import sys
from pathlib import Path


def test_comment_counts_failures_and_errors_and_links_report(tmp_path):
    report = tmp_path / "report.xml"
    report.write_text(
        '<testsuites><testsuite tests="7" failures="2" errors="1" skipped="1" /></testsuites>',
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[3] / ".github/scripts/functional_ci_comment.py"
    result = subprocess.run(
        [sys.executable, str(script), str(report), "https://github.com/org/repo/actions/runs/123/artifacts/456"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == (
        "Functional CI: 3 failed / 7 tests. "
        "[Feature matrix report](https://github.com/org/repo/actions/runs/123/artifacts/456)"
    )
