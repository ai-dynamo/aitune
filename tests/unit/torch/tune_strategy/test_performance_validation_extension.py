# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PerformanceValidationMixin."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from aitune.torch.backend.backend import Backend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.tune_strategy.mixin.performance_validation_mixin import (
    PerformanceValidationMixin,
    PerformanceValidationMixinConfig,
)

_PATCH_FIND_MAX_THROUGHPUT = (
    "aitune.torch.tune_strategy.mixin.performance_validation_mixin.find_max_throughput_for_backend"
)


class _ConcreteExtension(PerformanceValidationMixin):
    """Minimal concrete subclass for unit testing (TuneStrategy is abstract)."""

    def _tune(self, module, name, graph_spec, data, device, cache_dir):
        raise NotImplementedError

    def _describe_parts(self):
        return []

    def to_json_dict(self) -> dict[str, Any]:
        return {}


@pytest.fixture
def mock_module():
    m = MagicMock(spec=nn.Module)
    m.to = MagicMock()
    return m


@pytest.fixture
def mock_graph_spec():
    gs = MagicMock(spec=GraphSpec)
    gs.name = "test_graph"
    gs.input_spec = MagicMock()
    gs.get_max_batch_size = MagicMock(return_value=4)
    return gs


@pytest.fixture
def mock_data():
    return [MagicMock(spec=Sample)]


@pytest.fixture
def mock_backend():
    b = MagicMock(spec=Backend)
    b.describe.return_value = "mock_backend"
    b.key.return_value = "mock_backend"
    b.__deepcopy__ = lambda _, memo=None: b
    b.build.return_value = b
    return b


@pytest.fixture
def mock_eager_backend():
    b = MagicMock(spec=Backend)
    b.describe.return_value = "TorchEagerBackend"
    b.build.return_value = b
    b.__deepcopy__ = lambda _, memo=None: b
    return b


# ── _pre_tune ──────────────────────────────────────────────────────────────


def test_pre_tune_sets_baseline_throughput(
    mock_module, mock_graph_spec, mock_data, mock_eager_backend, torch_device, tmp_path
):
    """_pre_tune builds TorchEager and sets _baseline_throughput."""
    ext = _ConcreteExtension()
    ext.enable_find_max_batch_size(False)

    with (
        patch(
            "aitune.torch.tune_strategy.mixin.performance_validation_mixin.TorchEagerBackend",
            return_value=mock_eager_backend,
        ),
        patch(_PATCH_FIND_MAX_THROUGHPUT, return_value=(4, 100.0, MagicMock())),
    ):
        ext._pre_tune(mock_module, "mod", mock_graph_spec, mock_data, torch_device, tmp_path)

    assert ext._baseline_throughput == 100.0
    assert ext._baseline_backend is mock_eager_backend


def test_pre_tune_stores_baseline_backend(
    mock_module, mock_graph_spec, mock_data, mock_eager_backend, torch_device, tmp_path
):
    """_pre_tune stores the built TorchEager backend for later fallback use."""
    ext = _ConcreteExtension()
    ext.enable_find_max_batch_size(False)

    with (
        patch(
            "aitune.torch.tune_strategy.mixin.performance_validation_mixin.TorchEagerBackend",
            return_value=mock_eager_backend,
        ),
        patch(_PATCH_FIND_MAX_THROUGHPUT, return_value=(4, 50.0, MagicMock())),
    ):
        ext._pre_tune(mock_module, "mod", mock_graph_spec, mock_data, torch_device, tmp_path)

    assert ext._baseline_backend is mock_eager_backend


def test_pre_tune_resets_results_on_each_call(
    mock_module, mock_graph_spec, mock_data, mock_eager_backend, torch_device, tmp_path
):
    """_pre_tune resets perf_validation_results so stale results don't persist."""
    ext = _ConcreteExtension()
    ext.enable_find_max_batch_size(False)
    ext.perf_validation_results = [MagicMock()]  # pre-populate to confirm reset

    with (
        patch(
            "aitune.torch.tune_strategy.mixin.performance_validation_mixin.TorchEagerBackend",
            return_value=mock_eager_backend,
        ),
        patch(_PATCH_FIND_MAX_THROUGHPUT, return_value=(4, 100.0, MagicMock())),
    ):
        ext._pre_tune(mock_module, "mod", mock_graph_spec, mock_data, torch_device, tmp_path)

    assert ext.perf_validation_results == []


def test_pre_tune_baseline_build_failure_leaves_baseline_none(
    mock_module, mock_graph_spec, mock_data, torch_device, tmp_path
):
    """When baseline TorchEager build fails, _baseline_throughput stays None (performance check skipped)."""
    ext = _ConcreteExtension()
    ext.enable_find_max_batch_size(False)

    failing_eager = MagicMock()
    failing_eager.build.side_effect = RuntimeError("build failed")

    with patch(
        "aitune.torch.tune_strategy.mixin.performance_validation_mixin.TorchEagerBackend",
        return_value=failing_eager,
    ):
        ext._pre_tune(mock_module, "mod", mock_graph_spec, mock_data, torch_device, tmp_path)

    assert ext._baseline_throughput is None
    assert ext._baseline_backend is None


# ── batch size resolution ─────────────────────────────────────────────────


def test_resolved_batch_size_uses_graph_spec_get_max_batch_size(
    mock_module, mock_graph_spec, mock_data, mock_eager_backend, torch_device, tmp_path
):
    """_resolved_batch_size is taken from graph_spec.get_max_batch_size() after super()._pre_tune()."""
    mock_graph_spec.get_max_batch_size.return_value = 16
    ext = _ConcreteExtension()
    ext.enable_find_max_batch_size(False)

    with (
        patch(
            "aitune.torch.tune_strategy.mixin.performance_validation_mixin.TorchEagerBackend",
            return_value=mock_eager_backend,
        ),
        patch(_PATCH_FIND_MAX_THROUGHPUT, return_value=(16, 100.0, MagicMock())),
    ):
        ext._pre_tune(mock_module, "mod", mock_graph_spec, mock_data, torch_device, tmp_path)

    assert ext._resolved_batch_size == 16


def test_pre_tune_always_profiles_baseline_regardless_of_validate_flag(
    mock_module, mock_graph_spec, mock_data, mock_eager_backend, torch_device, tmp_path
):
    """_pre_tune always profiles TorchEager baseline even when validate_against_baseline=False."""
    ext = _ConcreteExtension()
    ext.enable_validate_against_baseline(False)
    ext.enable_find_max_batch_size(False)

    with (
        patch(
            "aitune.torch.tune_strategy.mixin.performance_validation_mixin.TorchEagerBackend",
            return_value=mock_eager_backend,
        ),
        patch(_PATCH_FIND_MAX_THROUGHPUT, return_value=(4, 80.0, MagicMock())) as mock_profile,
    ):
        ext._pre_tune(mock_module, "mod", mock_graph_spec, mock_data, torch_device, tmp_path)

    mock_profile.assert_called_once()
    assert ext._baseline_throughput == 80.0
    assert ext._resolved_batch_size == mock_graph_spec.get_max_batch_size.return_value


# ── _build_validate_and_check_perf ────────────────────────────────────────


def test_check_perf_returns_none_when_correctness_fails(
    mock_module, mock_graph_spec, mock_data, mock_backend, torch_device, tmp_path
):
    """Returns None immediately when _build_and_validate_backend fails (correctness)."""
    ext = _ConcreteExtension()
    ext._baseline_throughput = 1.0
    ext._resolved_batch_size = 4

    with patch.object(ext, "_build_and_validate_backend", return_value=None):
        result = ext._build_validate_and_check_perf(
            mock_backend, mock_module, "mod", mock_graph_spec, mock_data, torch_device, tmp_path
        )

    assert result is None
    assert ext.perf_validation_results == []


def test_check_perf_appends_result_and_returns_backend_when_passing(
    mock_module, mock_graph_spec, mock_data, mock_backend, torch_device, tmp_path
):
    """Appends PerformanceValidationMixinResult(passed=True) and returns backend when speedup meets threshold."""
    ext = _ConcreteExtension()
    ext._baseline_throughput = 1.0  # baseline: 1 sample/s
    ext._resolved_batch_size = 4

    with (
        patch.object(ext, "_build_and_validate_backend", return_value=mock_backend),
        patch(_PATCH_FIND_MAX_THROUGHPUT, return_value=(4, 2.0, MagicMock())),  # backend: 2 samples/s → speedup 2.0
    ):
        result = ext._build_validate_and_check_perf(
            mock_backend, mock_module, "mod", mock_graph_spec, mock_data, torch_device, tmp_path
        )

    assert result is mock_backend
    assert len(ext.perf_validation_results) == 1
    r = ext.perf_validation_results[0]
    assert r.throughput == 2.0
    assert r.baseline_throughput == 1.0
    assert abs(r.speedup - 2.0) < 0.01
    assert r.passed is True


def test_check_perf_appends_result_and_returns_none_when_gate_rejects(
    mock_module, mock_graph_spec, mock_data, mock_backend, torch_device, tmp_path
):
    """Returns None and records passed=False when speedup < threshold (gate enabled)."""
    ext = _ConcreteExtension()
    ext._baseline_throughput = 2.0  # baseline: 2 samples/s
    ext._resolved_batch_size = 4

    with (
        patch.object(ext, "_build_and_validate_backend", return_value=mock_backend),
        patch(_PATCH_FIND_MAX_THROUGHPUT, return_value=(4, 1.0, MagicMock())),  # backend: 1 sample/s → speedup 0.5
    ):
        result = ext._build_validate_and_check_perf(
            mock_backend, mock_module, "mod", mock_graph_spec, mock_data, torch_device, tmp_path
        )

    assert result is None
    assert len(ext.perf_validation_results) == 1
    assert ext.perf_validation_results[0].passed is False


def test_check_perf_returns_backend_when_gate_disabled_but_slow(
    mock_module, mock_graph_spec, mock_data, mock_backend, torch_device, tmp_path
):
    """When validate_against_baseline=False, slow backend is still returned."""
    ext = _ConcreteExtension()
    ext.enable_validate_against_baseline(False)
    ext._baseline_throughput = 2.0
    ext._resolved_batch_size = 4

    with (
        patch.object(ext, "_build_and_validate_backend", return_value=mock_backend),
        patch(_PATCH_FIND_MAX_THROUGHPUT, return_value=(4, 1.0, MagicMock())),  # slow
    ):
        result = ext._build_validate_and_check_perf(
            mock_backend, mock_module, "mod", mock_graph_spec, mock_data, torch_device, tmp_path
        )

    assert result is mock_backend
    assert ext.perf_validation_results[0].passed is False  # recorded but gate skipped


def test_check_perf_returns_backend_when_no_baseline(
    mock_module, mock_graph_spec, mock_data, mock_backend, torch_device, tmp_path
):
    """When _baseline_throughput is None (baseline failed), gate never applies."""
    ext = _ConcreteExtension()
    ext._baseline_throughput = None  # simulates baseline build failure
    ext._resolved_batch_size = 4

    with patch.object(ext, "_build_and_validate_backend", return_value=mock_backend):
        result = ext._build_validate_and_check_perf(
            mock_backend, mock_module, "mod", mock_graph_spec, mock_data, torch_device, tmp_path
        )

    assert result is mock_backend
    assert ext.perf_validation_results == []


def test_check_perf_returns_backend_when_profiling_fails(
    mock_module, mock_graph_spec, mock_data, mock_backend, torch_device, tmp_path
):
    """When candidate profiling raises, backend is accepted and no result recorded."""
    ext = _ConcreteExtension()
    ext._baseline_throughput = 1.0
    ext._resolved_batch_size = 4

    with (
        patch.object(ext, "_build_and_validate_backend", return_value=mock_backend),
        patch(_PATCH_FIND_MAX_THROUGHPUT, side_effect=RuntimeError("profiling failed")),
    ):
        result = ext._build_validate_and_check_perf(
            mock_backend, mock_module, "mod", mock_graph_spec, mock_data, torch_device, tmp_path
        )

    assert result is mock_backend
    assert ext.perf_validation_results == []


# ── PerformanceValidationMixinConfig ───────────────────────────────────────────


def test_perf_validation_config_defaults():
    """Default config uses 1% threshold and no explicit profiling config."""
    config = PerformanceValidationMixinConfig()
    assert config.min_speedup_threshold == 0.01
    assert config.profiling_config is None


def test_perf_validation_config_custom_threshold():
    config = PerformanceValidationMixinConfig(min_speedup_threshold=0.10)
    assert config.min_speedup_threshold == 0.10


def test_profiling_config_for_batch_size_default_uses_stable_window():
    """Default profiling config uses StableWindowMeasuringStopStrategy with global defaults."""
    from aitune.torch.config import DEFAULT_STABILITY_PERCENTAGE, DEFAULT_WINDOW_SIZE
    from aitune.torch.task.profiling import StableWindowMeasuringStopStrategy

    config = PerformanceValidationMixinConfig()
    profiling_cfg = config.profiling_config_for_batch_size(8)

    assert profiling_cfg.batch_sizes == [8]
    assert profiling_cfg.batching is True
    assert isinstance(profiling_cfg.measurement_stop_strategy, StableWindowMeasuringStopStrategy)
    assert profiling_cfg.measurement_stop_strategy.window_size == DEFAULT_WINDOW_SIZE
    assert profiling_cfg.measurement_stop_strategy.stability_percentage == DEFAULT_STABILITY_PERCENTAGE


def test_profiling_config_for_batch_size_user_override_replaces_batch_sizes():
    """When profiling_config is provided, only batch_sizes is overridden."""
    from aitune.torch.task.profiling import ProfilingConfig

    user_cfg = ProfilingConfig(batch_sizes=[1, 2, 4])
    config = PerformanceValidationMixinConfig(profiling_config=user_cfg)
    result = config.profiling_config_for_batch_size(16)

    assert result.batch_sizes == [16]


# ── _post_tune ────────────────────────────────────────────────────────────


def test_post_tune_emits_warning_when_logger_below_info(mock_backend, mock_graph_spec, mock_data):
    """At WARNING level (not INFO-enabled), emits full format via logger.warning."""
    from unittest.mock import patch

    ext = _ConcreteExtension()
    ext.perf_validation_results = [
        _make_result(mock_backend.describe(), throughput=200.0, baseline=100.0, speedup=2.0),
    ]

    with (
        patch.object(ext._logger, "isEnabledFor", return_value=False),
        patch.object(ext._logger, "warning") as mock_warn,
    ):
        ext._post_tune(mock_backend, "my_module", mock_graph_spec, mock_data)

    mock_warn.assert_called_once()
    msg = mock_warn.call_args[0][0]
    assert "speedup: 2.00x" in msg
    assert "100.00 → 200.00 samples/s" in msg
    assert "my_module" in msg
    assert mock_backend.describe() in msg


def test_post_tune_emits_short_format_via_sink_when_info_enabled(mock_backend, mock_graph_spec, mock_data):
    """At INFO level, emits short format via _sink (no module/backend fields)."""
    from unittest.mock import MagicMock, patch

    sink = MagicMock()
    ext = _ConcreteExtension(sink=sink)
    ext.perf_validation_results = [
        _make_result(mock_backend.describe(), throughput=200.0, baseline=100.0, speedup=2.0),
    ]

    with patch.object(ext._logger, "isEnabledFor", return_value=True):
        ext._post_tune(mock_backend, "my_module", mock_graph_spec, mock_data)

    sink.assert_called_once()
    msg = sink.call_args[0][0]
    assert "⚡ speedup: 2.00x" in msg
    assert "100.00 → 200.00 samples/s" in msg
    assert "my_module" not in msg


def test_post_tune_silent_when_backend_is_none(mock_graph_spec, mock_data):
    """No output when backend is None (tuning failed)."""
    from unittest.mock import patch

    ext = _ConcreteExtension()
    with patch.object(ext._logger, "warning") as mock_warn:
        ext._post_tune(None, "my_module", mock_graph_spec, mock_data)

    mock_warn.assert_not_called()


def test_post_tune_silent_when_backend_not_in_results(mock_backend, mock_graph_spec, mock_data):
    """No output when selected backend has no profiling result (e.g. OneBackendStrategy fallback)."""
    from unittest.mock import patch

    ext = _ConcreteExtension()
    ext.perf_validation_results = []  # empty — no match

    with (
        patch.object(ext._logger, "warning") as mock_warn,
        patch.object(ext._logger, "isEnabledFor", return_value=False),
    ):
        ext._post_tune(mock_backend, "my_module", mock_graph_spec, mock_data)

    mock_warn.assert_not_called()


def test_post_tune_emitted_when_validate_against_baseline_false(mock_backend, mock_graph_spec, mock_data):
    """Summary is emitted even when validate_against_baseline=False (profiling is unconditional)."""
    from unittest.mock import patch

    ext = _ConcreteExtension()
    ext.enable_validate_against_baseline(False)
    ext.perf_validation_results = [
        _make_result(mock_backend.describe(), throughput=150.0, baseline=100.0, speedup=1.5),
    ]

    with (
        patch.object(ext._logger, "isEnabledFor", return_value=False),
        patch.object(ext._logger, "warning") as mock_warn,
    ):
        ext._post_tune(mock_backend, "mod", mock_graph_spec, mock_data)

    mock_warn.assert_called_once()
    assert "speedup: 1.50x" in mock_warn.call_args[0][0]


def _make_result(description, *, throughput, baseline, speedup):
    from aitune.torch.tune_strategy.mixin.performance_validation_mixin import PerformanceValidationMixinResult

    return PerformanceValidationMixinResult(
        backend_description=description,
        throughput=throughput,
        baseline_throughput=baseline,
        speedup=speedup,
        passed=speedup >= 1.05,
    )


def test_extension_classes_exported_from_package():
    """New classes are accessible from the extension package."""
    from aitune.torch.tune_strategy.mixin import (
        PerformanceValidationMixin,
        PerformanceValidationMixinConfig,
        PerformanceValidationMixinResult,
    )

    assert PerformanceValidationMixinConfig is not None
    assert PerformanceValidationMixinResult is not None
    assert PerformanceValidationMixin is not None
