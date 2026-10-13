# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise subprocess report transport with real pytest consumers and no GPU."""

import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

import pytest


@pytest.fixture
def isolated_project(tmp_path):
    (tmp_path / "conftest.py").write_text('pytest_plugins = ["jit_process"]\n')
    # CLI logging raises verbosity even with -q, matching the repository's CI settings.
    (tmp_path / "pytest.ini").write_text("[pytest]\nlog_cli = true\njunit_logging = all\njunit_family = xunit1\n")
    # Split the selector so this regression file itself remains an ordinary test.
    (tmp_path / "test_reports.py").write_text(
        "# "
        "Patch"
        "er\n"
        + """import os
import pytest

state = []

class TestGPU:
    @pytest.fixture(autouse=True)
    def require_cuda(self):
        pytest.skip("CUDA unavailable in class fixture")

    def test_needs_cuda(self):
        pytest.fail("a skipped GPU test must not execute")

@pytest.mark.skip(reason="optional GPU backend")
def test_marked_skip():
    pytest.fail("a marked skip must not execute")

def test_runtime_skip():
    pytest.skip("runtime device unavailable")

@pytest.mark.xfail(reason="known backend failure")
def test_expected_failure():
    assert False, "expected failure"

def test_pass(record_property):
    assert os.environ.get("AITUNE_JIT_TEST_REPORTS")
    assert not state
    state.append(1)
    record_property("source", "spawned worker")

def test_failure():
    print("child failure diagnostic")
    assert False, "backend assertion preserved"

@pytest.fixture
def setup_error():
    raise ValueError("setup error preserved")

def test_setup_error(setup_error):
    pass

@pytest.fixture
def teardown_error():
    yield
    raise ValueError("teardown error preserved")

def test_teardown_error(teardown_error):
    pass
"""
    )
    return tmp_path


def run_isolated(project, workers, *options):
    env = dict(os.environ)
    env.pop("AITUNE_JIT_TEST_REPORTS", None)
    env["PYTEST_ADDOPTS"] = ""
    helper_dir = Path(__file__).resolve().parents[2] / "utilities"
    env["PYTHONPATH"] = os.pathsep.join([str(helper_dir), env.get("PYTHONPATH", "")])
    command = [sys.executable, "-m", "pytest", "-q", *options, "--junitxml=report.xml"]
    if workers:
        command.extend(["-n", str(workers)])
    result = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True, timeout=120)
    (project / "pytest.log").write_text(result.stdout + result.stderr)
    return result


@pytest.mark.parametrize("workers", [0, 2], ids=["serial", "xdist"])
@pytest.mark.parametrize("summary", ["-rf", "-ra", "-rs"])
def test_spawned_reports_preserve_skips_failures_and_junit(isolated_project, workers, summary):
    result = run_isolated(isolated_project, workers, summary)
    output = result.stdout + result.stderr
    assert result.returncode == 1, output  # Deliberate failures must remain normal pytest failures.
    assert "INTERNALERROR" not in output
    assert "1 xfailed" in output
    cases = {case.attrib["name"]: case for case in ElementTree.parse(isolated_project / "report.xml").iter("testcase")}
    for name, reason in (
        ("test_needs_cuda", "CUDA unavailable in class fixture"),
        ("test_marked_skip", "optional GPU backend"),
        ("test_runtime_skip", "runtime device unavailable"),
        ("test_expected_failure", "known backend failure"),
    ):
        skipped = cases[name].find("skipped")
        assert skipped is not None
        assert reason in skipped.attrib["message"]
    assert cases["test_failure"].find("failure") is not None
    assert "backend assertion preserved" in ElementTree.tostring(cases["test_failure"], encoding="unicode")
    assert "child failure diagnostic" in ElementTree.tostring(cases["test_failure"], encoding="unicode")
    for name in ("test_setup_error", "test_teardown_error"):
        assert cases[name].find("error") is not None
        assert name.removeprefix("test_").replace("_", " ") + " preserved" in ElementTree.tostring(
            cases[name], encoding="unicode"
        )
    property_element = cases["test_pass"].find("properties/property")
    assert property_element is not None
    assert property_element.attrib == {"name": "source", "value": "spawned worker"}


@pytest.mark.parametrize("workers", [0, 2], ids=["serial", "xdist"])
def test_worker_exit_without_call_report_remains_a_test_failure(isolated_project, workers):
    (isolated_project / "test_reports.py").write_text(
        "# "
        "Patch"
        "er\n"
        + """import os
def test_abrupt_exit(capfd):
    with capfd.disabled():
        print("abrupt child exit diagnostic", flush=True)
        os._exit(7)
"""
    )
    result = run_isolated(isolated_project, workers, "-ra")
    assert result.returncode == 1, result.stdout + result.stderr
    case = next(ElementTree.parse(isolated_project / "report.xml").iter("testcase"))
    assert case.find("failure") is not None
    assert "abrupt child exit diagnostic" in ElementTree.tostring(case, encoding="unicode")


@pytest.mark.parametrize("workers", [0, 1], ids=["serial", "xdist"])
@pytest.mark.parametrize("finalizer_outcome", ["normal", "error", "skip", "both-errors"])
def test_ordinary_isolated_ordinary_boundary_finalizes_collectors(
    isolated_project, workers, finalizer_outcome, plugin_options=()
):
    (isolated_project / "test_reports.py").unlink()
    (isolated_project / "conftest.py").write_text("""import os
from pathlib import Path
import pytest

pytest_plugins = ["jit_process"]

def lifetime(request):
    key = f"{os.getpid()}:{request.scope}:{request.node.nodeid}"
    path = Path(__file__).with_name("events.txt")
    with path.open("a") as stream:
        stream.write(f"setup:{key}\\n")
    yield
    with path.open("a") as stream:
        stream.write(f"teardown:{key}\\n")

@pytest.fixture(scope="session", autouse=True)
def session_lifetime(request):
    yield from lifetime(request)

@pytest.fixture(scope="module", autouse=True)
def module_lifetime(request):
    yield from lifetime(request)

@pytest.fixture(scope="class", autouse=True)
def class_lifetime(request):
    yield from lifetime(request)

@pytest.fixture(autouse=True)
def function_lifetime(request):
    yield from lifetime(request)
""")
    for package in ("a", "b"):
        (isolated_project / package).mkdir()
        (isolated_project / package / "__init__.py").touch()
    (isolated_project / "a/conftest.py").write_text(f"""import os
import pytest
from conftest import lifetime

@pytest.fixture(scope="package", autouse=True)
def package_lifetime(request):
    yield from lifetime(request)
    if not os.environ.get("AITUNE_JIT_TEST_REPORTS"):
        if {finalizer_outcome!r} in ("error", "both-errors"):
            raise RuntimeError("parent package finalizer failed")
        if {finalizer_outcome!r} == "skip":
            pytest.skip("parent package finalizer skipped")
""")
    for path, name, isolated in (
        ("a/test_before.py", "before", False),
        ("a/test_isolated.py", "isolated", True),
        ("b/test_after.py", "after", False),
    ):
        selector = "# Patch{}\n".format("er") if isolated else ""
        (isolated_project / path).write_text(
            selector
            + f"""import os
class Test{name.title()}:
    def test_{name}(self):
        assert bool(os.environ.get("AITUNE_JIT_TEST_REPORTS")) is {isolated!r}
"""
        )
    if finalizer_outcome == "both-errors":
        with (isolated_project / "a/test_isolated.py").open("a") as stream:
            stream.write("""import pytest
@pytest.fixture(autouse=True)
def child_cleanup():
    yield
    raise RuntimeError("child finalizer failed")
""")
    result = run_isolated(isolated_project, workers, "-ra", *plugin_options)
    output = result.stdout + result.stderr
    error_count = {"normal": 0, "error": 1, "skip": 0, "both-errors": 2}[finalizer_outcome]
    assert result.returncode == bool(error_count), output
    assert "previous item was not torn down properly" not in output
    assert "INTERNALERROR" not in output
    cases = list(ElementTree.parse(isolated_project / "report.xml").iter("testcase"))
    assert len(cases) == 3 + (finalizer_outcome == "both-errors")
    assert {case.attrib["name"] for case in cases} == {"test_before", "test_isolated", "test_after"}
    errors = [case for case in cases if case.find("error") is not None]
    assert len(errors) == error_count
    if errors:
        assert all(case.attrib["name"] == "test_isolated" for case in errors)
        assert "parent package finalizer failed" in ElementTree.tostring(errors[-1], encoding="unicode")
    if finalizer_outcome == "both-errors":
        assert "child finalizer failed" in ElementTree.tostring(errors[0], encoding="unicode")
    skipped = [case for case in cases if case.find("skipped") is not None]
    assert len(skipped) == (finalizer_outcome == "skip")
    if skipped:
        assert "parent package finalizer skipped" in ElementTree.tostring(skipped[0], encoding="unicode")
    events = (isolated_project / "events.txt").read_text().splitlines()
    assert Counter(event.removeprefix("setup:") for event in events if event.startswith("setup:")) == Counter(
        event.removeprefix("teardown:") for event in events if event.startswith("teardown:")
    )


@pytest.mark.parametrize("workers", [0, 1], ids=["serial", "xdist"])
@pytest.mark.parametrize("finalizer_outcome", ["normal", "error"])
def test_rerun_plugin_does_not_customize_uninitialized_parent_report(isolated_project, workers, finalizer_outcome):
    pytest.importorskip("pytest_rerunfailures")
    test_ordinary_isolated_ordinary_boundary_finalizes_collectors(
        isolated_project, workers, finalizer_outcome, plugin_options=("-p", "rerunfailures")
    )


@pytest.mark.parametrize("workers", [0, 1], ids=["serial", "xdist"])
@pytest.mark.parametrize("finalizer_outcome", ["normal", "error"])
def test_stateful_report_hooks_run_only_for_initialized_items(
    isolated_project, workers, finalizer_outcome, monkeypatch
):
    # Model a report-hook setup prerequisite; this is not the external rerun plugin.
    (isolated_project / "stateful_report_plugin.py").write_text("""import pytest

@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if report.when == "setup":
        item._terminal_errors = {}
    item._terminal_errors[report.when] = report.failed
    report.user_properties.append(("stateful_report_hook", "initialized"))
    return report
""")
    monkeypatch.setenv("PYTEST_PLUGINS", "stateful_report_plugin")
    test_ordinary_isolated_ordinary_boundary_finalizes_collectors(isolated_project, workers, finalizer_outcome)
    for case in ElementTree.parse(isolated_project / "report.xml").iter("testcase"):
        assert any(
            prop.attrib == {"name": "stateful_report_hook", "value": "initialized"}
            for prop in case.findall("properties/property")
        )
