# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the report-building reporting."""

import json
from importlib import import_module
from unittest.mock import MagicMock

import pytest

from aitune.__version__ import __version__
from aitune.torch.config import DEFAULT_TUNING_DATA_OUTPUT_PATH, AITuneConfig, AITuneMode
from aitune.torch.dynamic_shapes import BatchDim
from aitune.torch.tune_data.report_models import SCHEMA_VERSION, ExceptionInfo, ModuleInspectionReport
from aitune.torch.tune_data.reporting import (
    _active_graph,
    _active_module,
    _active_report,
    report_backend_build,
    report_graph_tune,
    report_inspection_details,
    report_module_tune,
    report_tune_run,
    report_tune_run_end,
    report_tune_run_start,
    snapshot_config,
    snapshot_tuning_data,
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
    """Patch config with a temporary output path."""
    report_path = tmp_path / "report.json"
    mock_config = mocker.patch("aitune.torch.tune_data.reporting.config")
    mock_config.tuning_data_output_path = report_path
    mocker.patch("aitune.torch.tune_data.reporting.snapshot_config", return_value={"mocked": True})
    return report_path


# ---------------------------------------------------------------------------
# snapshot_config
# ---------------------------------------------------------------------------


def test_snapshot_config_declarative_mode(mocker):
    # given — real AITuneConfig so to_dict() works
    real_config = AITuneConfig()
    real_config.min_num_samples = 50
    real_config.max_num_samples_stored = 2
    real_config.strict_mode = False
    mocker.patch("aitune.torch.tune_data.reporting.config", real_config)

    # when
    result = snapshot_config(AITuneMode.DECLARATIVE)

    # then — every public attribute of AITuneConfig appears, under its public name
    assert result["min_num_samples"] == 50
    assert result["max_num_samples_stored"] == 2
    assert result["strict_mode"] is False
    assert result["enable_diffusers_integration"] is True
    assert result["enable_transformers_integration"] is True
    assert result["device_after_tuning"] == "meta"
    assert "cache_dir" in result
    assert "tuning_data_output_path" in result
    assert result["tuning_data_output_path"] == str(real_config.tuning_data_output_path)
    assert not any(k.startswith("_") for k in result)


def test_snapshot_config_jit_mode(mocker):
    # given — real JIT Config so dataclasses.fields() works
    from aitune.torch.jit.config import Config as JitConfig

    jit_config = JitConfig()
    jit_config.min_samples = 10
    jit_config.max_depth_level = 3
    jit_config.min_parameters = 1000
    jit_config.skip_modules = ["Foo", "Bar"]
    mocker.patch("aitune.torch.tune_data.reporting.config")
    mocker.patch("aitune.torch.jit.config.config", jit_config)

    # when
    result = snapshot_config(AITuneMode.JIT)

    # then — every dataclass field appears, including ones previously dropped
    assert result["mode"] == "tune_eager"
    assert result["min_samples"] == 10
    assert result["max_depth_level"] == 3
    assert result["min_parameters"] == 1000
    assert result["skip_modules"] == ["Foo", "Bar"]
    # previously excluded fields are now captured
    assert "dry_run" in result
    assert "device" in result


def test_snapshot_config_jit_mode_strategy_resolves_default_when_unset(mocker):
    # given — default JIT config has no explicit strategy set
    from aitune.torch.jit.config import Config as JitConfig

    jit_config = JitConfig()
    mocker.patch("aitune.torch.tune_data.reporting.config")
    mocker.patch("aitune.torch.jit.config.config", jit_config)

    # when
    result = snapshot_config(AITuneMode.JIT)

    # then — snapshot reflects the resolved default (FirstWinsStrategy), not the sentinel
    assert result["strategy"]["name"] == "FirstWinsStrategy"
    assert "backends" in result["strategy"]["config"]


def test_snapshot_config_jit_mode_describes_strategy(mocker):
    # given
    from aitune.torch.jit.config import Config as JitConfig
    from aitune.torch.tune_strategy.tune_strategy import DummyTuneStrategy

    jit_config = JitConfig()
    jit_config.strategy = DummyTuneStrategy()
    mocker.patch("aitune.torch.tune_data.reporting.config")
    mocker.patch("aitune.torch.jit.config.config", jit_config)

    # when
    result = snapshot_config(AITuneMode.JIT)

    # then — strategy serialized via class name + to_json_dict, not raw repr
    assert result["strategy"] == {"name": "DummyTuneStrategy", "config": {}}


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
    info = ExceptionInfo(type="ValueError", message="bad")
    assert json_serialize({"exc": info}) == {"exc": {"type": "ValueError", "message": "bad"}}


def test_exception_info_limits_large_exception_payloads():
    message = "failure context\n" + "graph node 🔥\n" * 10_000 + "root cause"
    info = ExceptionInfo.from_exception(RuntimeError(message))

    assert info.message.startswith("failure context")
    assert info.message.endswith("root cause")
    assert "truncated" in info.message
    assert len(info.message.encode("utf-8")) <= 1024


# ---------------------------------------------------------------------------
# tune_run context manager
# ---------------------------------------------------------------------------


def test_tune_run_builds_and_flushes_report(enable_reporting):
    # when
    with report_tune_run(AITuneMode.JIT):
        assert _active_report.get() is not None

    # then
    report = json.loads(enable_reporting.read_text())
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
    report = json.loads(enable_reporting.read_text())
    assert report["exception"] == {"type": "RuntimeError", "message": "boom"}


# ---------------------------------------------------------------------------
# report_tune_run_start / report_tune_run_end (JIT path)
# ---------------------------------------------------------------------------


def test_start_end_tune_run(enable_reporting):
    # when
    report_tune_run_start(AITuneMode.JIT)
    assert _active_report.get() is not None
    report_tune_run_end()

    # then
    report = json.loads(enable_reporting.read_text())
    assert report["mode"] == "JIT"
    assert _active_report.get() is None


def test_end_tune_run_with_exception(enable_reporting):
    # when
    report_tune_run_start(AITuneMode.JIT)
    report_tune_run_end(exception=ValueError("bad"))

    # then
    report = json.loads(enable_reporting.read_text())
    assert report["exception"]["type"] == "ValueError"


# ---------------------------------------------------------------------------
# Full hierarchy
# ---------------------------------------------------------------------------


def _make_graph_spec(name="graph_0"):
    spec = MagicMock()
    spec.name = name
    spec.dynamic_shapes = None
    spec.input_spec.to_json_dict.return_value = {"tensor_data": [{"name": "x"}]}
    spec.output_spec.to_json_dict.return_value = {"tensor_data": [{"name": "y"}]}
    return spec


def test_graph_dynamic_shapes_are_json_serializable(enable_reporting):
    graph_spec = _make_graph_spec()
    graph_spec.dynamic_shapes = {
        ("options", "mask"): (BatchDim("batch", min=1, opt=2, max=4), 128),
    }

    with report_tune_run(AITuneMode.DECLARATIVE):
        with report_module_tune(module_name="encoder", num_parameters=100):
            with report_graph_tune(graph_spec, _make_strategy()):
                pass

    graph = json.loads(enable_reporting.read_text())["modules"][0]["graphs"][0]
    assert graph["dynamic_shapes"] == [
        {
            "path": ["options", "mask"],
            "shape": [
                {"type": "BatchDim", "name": "batch", "min": 1, "max": 4, "opt": 2},
                128,
            ],
        }
    ]


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
        with report_module_tune(module_name="encoder", num_parameters=5000, module_id=42):
            with report_graph_tune(_make_graph_spec(), _make_strategy()) as gt:
                with report_backend_build(_make_backend(config_dict={"precision": "fp16"})):
                    pass
                gt["selected_backend"] = "TensorRT(dynamo=True)"
                gt["strategy_results"] = [{"backend": "TRT", "success": True}]

    # then
    report = json.loads(enable_reporting.read_text())
    assert report["duration_s"] is not None
    assert report["exception"] is None

    assert len(report["modules"]) == 1
    module = report["modules"][0]
    assert module["module_name"] == "encoder"
    assert module["module_id"] == 42
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
# Inspection details
# ---------------------------------------------------------------------------


def test_report_inspection_details_updates_active_report_without_flushing(enable_reporting):
    # given
    report_tune_run_start(AITuneMode.JIT)
    details = [
        ModuleInspectionReport(
            module_id=1,
            module_name="module",
            module_class="torch.nn.Module",
            state="recording",
            level=0,
            call_count=1,
            num_parameters=10,
            allowed_to_tune=True,
        )
    ]

    # when
    report_inspection_details(details)

    # then — active report is updated in memory, but disk flush is explicit
    report = _active_report.get()
    assert report is not None
    assert report.inspection_details == details
    assert not enable_reporting.exists()

    report_tune_run_end()


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
    report = json.loads(enable_reporting.read_text())
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
    graph = json.loads(enable_reporting.read_text())["modules"][0]["graphs"][0]
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
    build = json.loads(enable_reporting.read_text())["modules"][0]["graphs"][0]["backend_builds"][0]
    assert build["success"] is False
    assert build["exception"]["type"] == "OSError"


def test_build_log_file_stored_in_report(enable_reporting, tmp_path):
    # when
    log_file = tmp_path / "trt" / "build.log"
    with report_tune_run(AITuneMode.JIT):
        with report_module_tune(module_name="enc", num_parameters=100):
            with report_graph_tune(_make_graph_spec(), _make_strategy()):
                with report_backend_build(_make_backend(), log_file=log_file):
                    pass

    # then
    build = json.loads(enable_reporting.read_text())["modules"][0]["graphs"][0]["backend_builds"][0]
    assert build["log_file"] == str(log_file)


def test_build_log_file_none_when_not_provided(enable_reporting):
    # when
    with report_tune_run(AITuneMode.JIT):
        with report_module_tune(module_name="enc", num_parameters=100):
            with report_graph_tune(_make_graph_spec(), _make_strategy()):
                with report_backend_build(_make_backend()):
                    pass

    # then
    build = json.loads(enable_reporting.read_text())["modules"][0]["graphs"][0]["backend_builds"][0]
    assert build["log_file"] is None


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
    report = json.loads(enable_reporting.read_text())
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
    builds = json.loads(enable_reporting.read_text())["modules"][0]["graphs"][0]["backend_builds"]
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
    report = json.loads(enable_reporting.read_text())
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


# ---------------------------------------------------------------------------
# snapshot_tuning_data
# ---------------------------------------------------------------------------


def test_snapshot_tuning_data_flushes_mid_run(enable_reporting):
    # given
    report_tune_run_start(AITuneMode.JIT)

    # when — flush before the run ends
    result = snapshot_tuning_data()

    # then — partial report written to disk, path returned
    assert result == enable_reporting
    report = json.loads(enable_reporting.read_text())
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["aitune_version"] == __version__
    assert report["mode"] == "JIT"
    assert report["duration_s"] is None

    report_tune_run_end()


def test_snapshot_tuning_data_refreshes_jit_config_on_flush(mocker, tmp_path):
    # given
    from aitune.torch.jit.config import Config as JitConfig
    from aitune.torch.jit.config import JITMode

    report_path = tmp_path / "report.json"
    mock_config = mocker.patch("aitune.torch.tune_data.reporting.config")
    mock_config.cache_dir = tmp_path / "cache"
    mock_config.tuning_data_output_path = report_path

    report_tune_run_start(AITuneMode.JIT)

    jit_config = JitConfig()
    jit_config.mode = JITMode.TUNE_DEFERRED
    jit_config.min_samples = 8
    jit_config.max_depth_level = 4
    mocker.patch("aitune.torch.jit.config.config", jit_config)

    # when
    result = snapshot_tuning_data()

    # then
    assert result == report_path
    report = json.loads(report_path.read_text())
    assert report["aitune_config"]["mode"] == "tune_deferred"
    assert report["aitune_config"]["min_samples"] == 8
    assert report["aitune_config"]["max_depth_level"] == 4

    report_tune_run_end()


def test_snapshot_tuning_data_custom_path(tmp_path):
    # given
    custom_path = tmp_path / "custom" / "report.json"
    report_tune_run_start(AITuneMode.JIT)

    # when
    result = snapshot_tuning_data(custom_path)

    # then — written to the caller-supplied path, returned path matches
    assert result == custom_path
    assert custom_path.exists()
    report = json.loads(custom_path.read_text())
    assert report["mode"] == "JIT"

    report_tune_run_end()


def test_snapshot_tuning_data_custom_path_is_not_rank_qualified(mocker, tmp_path):
    # given
    custom_path = tmp_path / "custom" / "report.json"
    distributed_output_path = mocker.patch("aitune.torch.tune_data.reporting.distributed_output_path")
    report_tune_run_start(AITuneMode.JIT)

    # when
    result = snapshot_tuning_data(custom_path)

    # then
    assert result == custom_path
    assert custom_path.exists()
    distributed_output_path.assert_not_called()

    report_tune_run_end()


def test_configured_tuning_data_path_is_not_rank_qualified(mocker, tmp_path):
    # given
    configured_path = tmp_path / "configured" / "report.json"
    real_config = AITuneConfig()
    real_config.tuning_data_output_path = configured_path
    mocker.patch("aitune.torch.tune_data.reporting.config", real_config)
    distributed_output_path = mocker.patch("aitune.torch.tune_data.reporting.distributed_output_path")
    report_tune_run_start(AITuneMode.JIT)

    # when
    result = snapshot_tuning_data()

    # then
    assert result == configured_path
    assert configured_path.exists()
    distributed_output_path.assert_not_called()

    report_tune_run_end()


def test_environment_tuning_data_path_is_not_rank_qualified(mocker, tmp_path):
    # given
    environment_path = tmp_path / "environment" / "report.json"
    config_module = import_module("aitune.torch.config")
    mocker.patch.object(config_module, "TUNING_DATA_PATH", str(environment_path))
    real_config = AITuneConfig()
    mocker.patch("aitune.torch.tune_data.reporting.config", real_config)
    distributed_output_path = mocker.patch("aitune.torch.tune_data.reporting.distributed_output_path")
    report_tune_run_start(AITuneMode.JIT)

    # when
    result = snapshot_tuning_data()

    # then
    assert result == environment_path
    assert environment_path.exists()
    distributed_output_path.assert_not_called()

    report_tune_run_end()


def test_default_tuning_data_path_is_rank_qualified(mocker, tmp_path):
    # given
    rank_path = tmp_path / "report.rank-2-of-4.json"
    config_module = import_module("aitune.torch.config")
    mocker.patch.object(config_module, "TUNING_DATA_PATH", None)
    real_config = AITuneConfig()
    mocker.patch("aitune.torch.tune_data.reporting.config", real_config)
    distributed_output_path = mocker.patch(
        "aitune.torch.tune_data.reporting.distributed_output_path", return_value=rank_path
    )
    report_tune_run_start(AITuneMode.JIT)

    # when
    result = snapshot_tuning_data()

    # then
    assert result == rank_path
    distributed_output_path.assert_called_with(DEFAULT_TUNING_DATA_OUTPUT_PATH)
    assert rank_path.exists()

    report_tune_run_end()


def test_snapshot_tuning_data_noop_without_active_run():
    # when — no run active
    result = snapshot_tuning_data()

    # then — returns None
    assert result is None


def test_second_run_overwrites_report(enable_reporting):
    # when — two sequential runs
    with report_tune_run(AITuneMode.JIT):
        pass
    with report_tune_run(AITuneMode.DECLARATIVE):
        pass

    # then — only the second run is in the file
    report = json.loads(enable_reporting.read_text())
    assert report["mode"] == "DECLARATIVE"


def test_flush_reraises_disk_space_error_on_enospc(enable_reporting, mocker):
    """Report flush must surface ENOSPC as DiskSpaceError instead of swallowing it at DEBUG."""
    import errno

    from aitune.utils.disk_space import DiskSpaceError

    report_tune_run_start(AITuneMode.JIT)

    # Force the JSON dump inside _flush_active_report to hit ENOSPC.
    mocker.patch(
        "aitune.torch.tune_data.reporting.json.dump",
        side_effect=OSError(errno.ENOSPC, "No space left on device"),
    )

    with pytest.raises(DiskSpaceError):
        report_tune_run_end()


def test_tune_run_start_checks_disk_space(mocker):
    """Starting a tune run runs a pre-flight disk-space check on the cache dir."""
    from aitune.torch.config import config as global_config

    check_mock = mocker.patch("aitune.torch.tune_data.reporting.check_disk_space")

    report_tune_run_start(AITuneMode.JIT)

    check_mock.assert_called_once_with(global_config.cache_dir)


def test_tune_run_start_checks_disk_space_even_without_reporting(mocker):
    """The pre-flight disk-space check runs even when tuning-data collection is disabled."""
    mock_config = mocker.patch("aitune.torch.tune_data.reporting.config")
    mock_config.enable_tuning_data_collection = False
    mock_config.cache_dir = "/some/cache"

    check_mock = mocker.patch("aitune.torch.tune_data.reporting.check_disk_space")

    report_tune_run_start(AITuneMode.JIT)

    check_mock.assert_called_once_with("/some/cache")


def test_flush_still_swallows_unrelated_errors(enable_reporting, mocker, caplog):
    """Non-ENOSPC report-flush failures remain best-effort (logged at DEBUG, not raised)."""
    import errno
    import logging

    report_tune_run_start(AITuneMode.JIT)

    mocker.patch(
        "aitune.torch.tune_data.reporting.json.dump",
        side_effect=OSError(errno.EACCES, "Permission denied"),
    )

    with caplog.at_level(logging.DEBUG, logger="aitune.torch.tune_data.reporting"):
        # Should NOT raise — reporting stays best-effort for unrelated errors.
        report_tune_run_end()

    assert any("Failed to write tuning report" in r.message for r in caplog.records)
