# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Format the functional CI PR comment from a JUnit report and artifact URL."""

import sys
import xml.etree.ElementTree as ET


def main() -> None:
    """Print functional test counts and link to the uploaded feature matrix."""
    if len(sys.argv) != 3:
        sys.exit("usage: functional_ci_comment.py report.xml artifact-url")
    root = ET.parse(sys.argv[1]).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    if not suites:
        sys.exit("no test suites in JUnit report")
    tests = sum(int(suite.get("tests", "0")) for suite in suites)
    failures = sum(int(suite.get("failures", "0")) + int(suite.get("errors", "0")) for suite in suites)
    skipped = sum(int(suite.get("skipped", "0")) for suite in suites)
    icon = ":white_check_mark:" if tests and not failures else ":x:"
    passed = tests - failures - skipped
    counts = f"{passed} passed, " + (f"{skipped} skipped, " if skipped else "") + f"{failures} failed"
    sys.stdout.write(f"{icon} Functional tests: {counts}\n\n[Feature matrix report]({sys.argv[2]})\n")


if __name__ == "__main__":
    main()
