# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Format the functional CI PR comment from a JUnit report and artifact URL."""

import sys
import xml.etree.ElementTree as ET


def main() -> None:
    """Print the failed-test count and link to the uploaded feature matrix."""
    if len(sys.argv) != 3:
        sys.exit("usage: functional_ci_comment.py report.xml artifact-url")
    root = ET.parse(sys.argv[1]).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    if not suites:
        sys.exit("no test suites in JUnit report")
    tests = sum(int(suite.get("tests", "0")) for suite in suites)
    failures = sum(int(suite.get("failures", "0")) + int(suite.get("errors", "0")) for suite in suites)
    sys.stdout.write(f"Functional CI: {failures} failed / {tests} tests. [Feature matrix report]({sys.argv[2]})\n")


if __name__ == "__main__":
    main()
