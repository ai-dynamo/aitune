# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for functional CI capabilities and reports."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


def test_missing_required_import_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.functional_ci import capabilities

    profile = capabilities.FunctionalProfile("test", 0, None, ("missing_package",), frozenset(), {})
    monkeypatch.setattr(capabilities, "_module_available", lambda name: False)

    with pytest.raises(RuntimeError, match="required dependency"):
        capabilities.check_environment(profile)


def test_missing_advertised_capability_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.functional_ci import capabilities

    profile = capabilities.FunctionalProfile("test", 0, None, (), frozenset({"fp8"}), {})
    monkeypatch.setattr(capabilities, "_capability_error", lambda profile, name: "unavailable")

    with pytest.raises(RuntimeError, match="fp8"):
        capabilities.check_environment(profile)


def test_optional_exclusion_has_identity() -> None:
    from tests.functional_ci.capabilities import FunctionalProfile, capability_reason
    from tests.functional_ci.reporting import case_properties

    profile = FunctionalProfile("test", 0, None, (), frozenset(), {"fp4": "not available in this runner"})
    assert capability_reason(profile, "fp4") == "not available in this runner"
    marker = pytest.mark.requires_capability(name="fp4").mark

    class Item:
        name = "test_example[fp4]"

        def iter_markers(self, name=None):
            return iter([marker] if name in (None, "requires_capability") else [])

    assert dict(case_properties(Item())) == {"case_id": "test_example[fp4]", "required_capability": "fp4"}


def test_failed_setup_retains_properties(pytester: pytest.Pytester) -> None:
    pytester.makeconftest('pytest_plugins = ["tests.functional_ci.conftest"]')
    pytester.makepyfile(
        """
        import pytest
        @pytest.fixture
        def broken():
            raise RuntimeError("setup failure")
        @pytest.mark.functional_case(backend="TorchEagerBackend", variant="failure")
        def test_case(broken):
            pass
        """
    )
    result = pytester.runpytest("--junitxml=report.xml")
    assert result.ret != 0
    properties = ET.parse("report.xml").find(".//testcase/properties")
    assert properties is not None
    assert {prop.attrib["name"]: prop.attrib["value"] for prop in properties}["backend"] == "TorchEagerBackend"
    assert ET.parse("report.xml").find(".//error") is not None


def test_duplicate_ids_are_collection_error(pytester: pytest.Pytester) -> None:
    pytester.makeconftest('pytest_plugins = ["tests.functional_ci.conftest"]')
    pytester.makepyfile(
        test_first="def test_example(): pass",
        test_second="def test_example(): pass",
    )
    result = pytester.runpytest("--collect-only", "-q")
    assert result.ret != 0
    assert "duplicate case ID" in result.stdout.str() + result.stderr.str()


def test_area_selection_uses_all_markers(pytester: pytest.Pytester) -> None:
    pytester.makeconftest('pytest_plugins = ["tests.functional_ci.conftest"]')
    pytester.makeini("""
        [pytest]
        markers =
            checkpoints: checkpoint coverage
            shapes: shape coverage
    """)
    pytester.makepyfile(
        """
        import pytest
        @pytest.mark.checkpoints
        @pytest.mark.shapes
        @pytest.mark.parametrize("variant", [pytest.param(1, id="dynamic-checkpoint")])
        def test_example(variant): pass
        @pytest.mark.checkpoints
        def test_other(): pass
        """
    )
    result = pytester.runpytest("--collect-only", "-q", "-m", "checkpoints and shapes")
    assert {line.strip().split("::")[-1] for line in result.stdout.str().splitlines() if "::" in line} == {
        "test_example[dynamic-checkpoint]"
    }


def test_run_metadata_has_environment() -> None:
    from tests.functional_ci.capabilities import PR_PROFILE
    from tests.functional_ci.reporting import collect_run_metadata

    metadata = collect_run_metadata(PR_PROFILE)
    required = {
        "schema_version",
        "commit_sha",
        "ci_run_id",
        "ci_run_attempt",
        "container_ref",
        "container_digest",
        "gpus",
        "gpu_count",
        "driver_version",
        "cuda_version",
        "library_versions",
        "capability_profile",
        "environment_errors",
    }
    assert required <= metadata.keys()


def test_metadata_survives_pytest_import_error(pytester: pytest.Pytester, tmp_path: Path) -> None:
    from tests.functional_ci.reporting import write_run_metadata

    metadata_path = tmp_path / "functional-ci.run.json"
    write_run_metadata(metadata_path, {"schema_version": 1})
    pytester.makepyfile("import missing_required_package")
    result = pytester.runpytest("--collect-only", "-q")
    assert result.ret != 0
    assert metadata_path.is_file()
    assert json.loads(metadata_path.read_text())["schema_version"] == 1
