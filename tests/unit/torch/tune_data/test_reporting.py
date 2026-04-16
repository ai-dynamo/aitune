# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the report-building reporting."""

import json
from unittest.mock import MagicMock

import pytest

from aitune.torch.config import AITuneMode
from aitune.torch.tune_data.report_models import ExceptionInfo
from aitune.torch.tune_data.reporting import (
    REPORT_FILENAME,
    _active_graph,
    _active_module,
    _active_report,
    report_backend_build,
    report_graph_tune,
    report_module_tune,
    report_tune_run,
    report_tune_run_end,
    report_tune_run_start,
    snapshot_config,
)
from aitune.utils.serialization import json_serialize


@pytest.fixture(autouse=True)
def _reset_contextvars():
    """Clear report context between tests."""
    _active_report.set(None)
    _active_module.set(None)
    _active_graph.set(None)
    yield
    _active_report.set(None)
    _active_module.set(None)
    _active_graph.set(None)


@pytest.fixture()
def enable_reporting(mocker, tmp_path):
    """Enable tuning data collection with a temporary output dir."""
    mock_config = mocker.patch("aitune.torch.tune_data.reporting.config")
    mock_config.enable_tuning_data_collection = True
    mock_config.tuning_data_output_dir = tmp_path
    mocker.patch("aitune.torch.tune_data.reporting.snapshot_config", return_value={"mocked": True})
    return tmp_path


# ---------------------------------------------------------------------------
# snapshot_config
# ---------------------------------------------------------------------------


def test_snapshot_config_declarative_mode(mocker):
    # given
    mock_config = mocker.patch("aitune.torch.tune_data.reporting.config")
    mock_config.min_num_samples = 100
    mock_config.max_num_samples_stored = 1
    mock_config.device_after_tuning = "meta"
    mock_config.strict_mode = True
    mock_config.enable_hf_integrations = False

    # when
    result = snapshot_config(AITuneMode.DECLARATIVE)

    # then
    assert result == {
        "min_num_samples": 100,
        "max_num_samples_stored": 1,
        "device_after_tuning": "meta",
        "strict_mode": True,
        "enable_hf_integrations": False,
    }


def test_snapshot_config_jit_mode(mocker):
    # given
    mocker.patch("aitune.torch.tune_data.reporting.config")
    mock_jit_config = mocker.patch("aitune.torch.jit.config.config")
    mock_jit_config.min_samples = 10
    mock_jit_config.batch_axis_required = True
    mock_jit_config.max_depth_level = 3
    mock_jit_config.min_parameters = 1000
    mock_jit_config.detect_graph_breaks = False
    mock_jit_config.skip_modules = []
    mock_jit_config.backends = []

    # when
    result = snapshot_config(AITuneMode.JIT)

    # then
    assert result == {
        "min_samples": 10,
        "batch_axis_required": True,
        "max_depth_level": 3,
        "min_parameters": 1000,
        "detect_graph_breaks": False,
        "skip_modules": [],
        "backends": [],
    }


def test_snapshot_config_invalid_mode_raises():
    with pytest.raises(ValueError, match="Invalid tuning mode"):
        snapshot_config("UNKNOWN")  # pytype: disable=wrong-arg-types


# ---------------------------------------------------------------------------
# json_serialize
# ---------------------------------------------------------------------------


def test_serialize_none():
    assert json_serialize(None) is None


def test_serialize_enum():
    assert json_serialize(AITuneMode.JIT) == "JIT"


def test_serialize_dataclass():
    info = ExceptionInfo(type="ValueError", message="bad", traceback="tb")
    assert json_serialize({"exc": info}) == {"exc": {"type": "ValueError", "message": "bad", "traceback": "tb"}}


# ---------------------------------------------------------------------------
# tune_run context manager
# ---------------------------------------------------------------------------


def test_tune_run_builds_and_flushes_report(enable_reporting):
    # when
    with report_tune_run(AITuneMode.JIT):
        assert _active_report.get() is not None

    # then
    report = json.loads((enable_reporting / REPORT_FILENAME).read_text())
    assert report["mode"] == "JIT"
    assert report["aitune_config"] == {"mocked": True}
    assert report["duration_s"] is not None
    assert report["exception"] is None
    assert _active_report.get() is None


def test_tune_run_captures_exception(enable_reporting):
    # when
    with pytest.raises(RuntimeError):
        with report_tune_run(AITuneMode.JIT):
            raise RuntimeError("boom")

    # then
    report = json.loads((enable_reporting / REPORT_FILENAME).read_text())
    assert report["exception"]["type"] == "RuntimeError"
    assert report["exception"]["message"] == "boom"


def test_tune_run_noop_when_disabled(mocker):
    # given
    mock_config = mocker.patch("aitune.torch.tune_data.reporting.config")
    mock_config.enable_tuning_data_collection = False

    # when
    with report_tune_run(AITuneMode.JIT):
        pass

    # then
    assert _active_report.get() is None


# ---------------------------------------------------------------------------
# report_tune_run_start / report_tune_run_end (JIT path)
# ---------------------------------------------------------------------------


def test_start_end_tune_run(enable_reporting):
    # when
    report_tune_run_start(AITuneMode.JIT)
    assert _active_report.get() is not None
    report_tune_run_end()

    # then
    report = json.loads((enable_reporting / REPORT_FILENAME).read_text())
    assert report["mode"] == "JIT"
    assert _active_report.get() is None


def test_end_tune_run_with_exception(enable_reporting):
    # when
    report_tune_run_start(AITuneMode.JIT)
    report_tune_run_end(exception=ValueError("bad"))

    # then
    report = json.loads((enable_reporting / REPORT_FILENAME).read_text())
    assert report["exception"]["type"] == "ValueError"


# ---------------------------------------------------------------------------
# Full hierarchy
# ---------------------------------------------------------------------------


def _make_graph_spec(name="graph_0"):
    spec = MagicMock()
    spec.name = name
    spec.input_spec.to_json_dict.return_value = {"tensor_data": [{"name": "x"}]}
    spec.output_spec.to_json_dict.return_value = {"tensor_data": [{"name": "y"}]}
    return spec


def _make_strategy(name="MaxThroughputStrategy"):
    strategy = MagicMock()
    strategy.__class__.__name__ = name
    strategy.to_json_dict.return_value = {"timeout": 60}
    return strategy


def _make_backend(describe="TensorRT(dynamo=True)", config_dict=None):
    be = MagicMock()
    be.describe.return_value = describe
    if config_dict is not None:
        be._config.to_dict.return_value = config_dict
        be._config._to_json.side_effect = lambda x: x
    else:
        be._config = None
    return be


def test_full_hierarchy_produces_nested_report(enable_reporting):
    # when
    with report_tune_run(AITuneMode.JIT):
        with report_module_tune(module_name="encoder", num_parameters=5000):
            with report_graph_tune(_make_graph_spec(), _make_strategy()) as gt:
                with report_backend_build(_make_backend(config_dict={"precision": "fp16"})):
                    pass
                gt["selected_backend"] = "TensorRT(dynamo=True)"
                gt["strategy_results"] = [{"backend": "TRT", "success": True}]

    # then
    report = json.loads((enable_reporting / REPORT_FILENAME).read_text())
    assert report["duration_s"] is not None
    assert report["exception"] is None

    assert len(report["modules"]) == 1
    module = report["modules"][0]
    assert module["module_name"] == "encoder"
    assert module["num_parameters"] == 5000
    assert module["duration_s"] is not None

    assert len(module["graphs"]) == 1
    graph = module["graphs"][0]
    assert graph["graph_name"] == "graph_0"
    assert graph["strategy_name"] == "MaxThroughputStrategy"
    assert graph["selected_backend"] == "TensorRT(dynamo=True)"
    assert len(graph["strategy_results"]) == 1

    assert len(graph["backend_builds"]) == 1
    build = graph["backend_builds"][0]
    assert build["backend"] == "TensorRT(dynamo=True)"
    assert build["backend_config"] == {"precision": "fp16"}
    assert build["success"] is True
    assert build["duration_s"] is not None


# ---------------------------------------------------------------------------
# Exceptions at each level
# ---------------------------------------------------------------------------


def test_module_level_exception(enable_reporting):
    # when
    with report_tune_run(AITuneMode.DECLARATIVE):
        with pytest.raises(ValueError):
            with report_module_tune(module_name="dec", num_parameters=100):
                raise ValueError("bad module")

    # then
    report = json.loads((enable_reporting / REPORT_FILENAME).read_text())
    module = report["modules"][0]
    assert module["exception"]["type"] == "ValueError"
    assert report["exception"] is None


def test_graph_level_exception(enable_reporting):
    # when
    with report_tune_run(AITuneMode.JIT):
        with report_module_tune(module_name="enc", num_parameters=100):
            with pytest.raises(TypeError):
                with report_graph_tune(_make_graph_spec(), _make_strategy()):
                    raise TypeError("bad graph")

    # then
    graph = json.loads((enable_reporting / REPORT_FILENAME).read_text())["modules"][0]["graphs"][0]
    assert graph["exception"]["type"] == "TypeError"


def test_build_level_exception(enable_reporting):
    # when
    with report_tune_run(AITuneMode.JIT):
        with report_module_tune(module_name="enc", num_parameters=100):
            with report_graph_tune(_make_graph_spec(), _make_strategy()):
                with pytest.raises(OSError):
                    with report_backend_build(_make_backend()):
                        raise OSError("build failed")

    # then
    build = json.loads((enable_reporting / REPORT_FILENAME).read_text())["modules"][0]["graphs"][0]["backend_builds"][0]
    assert build["success"] is False
    assert build["exception"]["type"] == "OSError"


# ---------------------------------------------------------------------------
# Multiple modules / graphs / backends
# ---------------------------------------------------------------------------


def test_multiple_modules_and_graphs_preserve_order(enable_reporting):
    # when
    with report_tune_run(AITuneMode.DECLARATIVE):
        for mod_name in ["mod_a", "mod_b"]:
            with report_module_tune(module_name=mod_name, num_parameters=100):
                for gname in ["g0", "g1"]:
                    with report_graph_tune(_make_graph_spec(gname), _make_strategy()):
                        pass

    # then
    report = json.loads((enable_reporting / REPORT_FILENAME).read_text())
    assert [m["module_name"] for m in report["modules"]] == ["mod_a", "mod_b"]
    assert [g["graph_name"] for g in report["modules"][0]["graphs"]] == ["g0", "g1"]
    assert [g["graph_name"] for g in report["modules"][1]["graphs"]] == ["g0", "g1"]


def test_multiple_backend_builds_per_graph(enable_reporting):
    # when
    with report_tune_run(AITuneMode.JIT):
        with report_module_tune(module_name="enc", num_parameters=100):
            with report_graph_tune(_make_graph_spec(), _make_strategy()) as gt:
                with report_backend_build(_make_backend(describe="TRT")):
                    pass
                with pytest.raises(RuntimeError):
                    with report_backend_build(_make_backend(describe="Inductor")):
                        raise RuntimeError("fail")
                gt["selected_backend"] = "TRT"

    # then
    builds = json.loads((enable_reporting / REPORT_FILENAME).read_text())["modules"][0]["graphs"][0]["backend_builds"]
    assert len(builds) == 2
    assert builds[0]["backend"] == "TRT"
    assert builds[0]["success"] is True
    assert builds[1]["backend"] == "Inductor"
    assert builds[1]["success"] is False


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------


def test_duration_computed(enable_reporting):
    # when
    with report_tune_run(AITuneMode.JIT):
        pass

    # then
    report = json.loads((enable_reporting / REPORT_FILENAME).read_text())
    assert report["duration_s"] >= 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_module_tune_noop_without_active_run():
    # when — no run active
    with report_module_tune(module_name="orphan", num_parameters=100):
        pass

    # then — no crash


def test_graph_tune_noop_without_active_module():
    # when — no module active
    with report_graph_tune(_make_graph_spec(), _make_strategy()) as gt:
        pass

    # then — yields empty dict, no crash
    assert gt == {}


def test_backend_build_noop_without_active_graph():
    # when — no graph active
    with report_backend_build(_make_backend()):
        pass

    # then — no crash


def test_second_run_overwrites_report(enable_reporting):
    # when — two sequential runs
    with report_tune_run(AITuneMode.JIT):
        pass
    with report_tune_run(AITuneMode.DECLARATIVE):
        pass

    # then — only the second run is in the file
    report = json.loads((enable_reporting / REPORT_FILENAME).read_text())
    assert report["mode"] == "DECLARATIVE"
