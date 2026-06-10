# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PerformanceValidationMixin."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from aitune.torch.backend.backend import Backend
from aitune.torch.backend.torch_eager import TorchEagerBackend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.module.wrapper_module import Module
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.task.profiling import (
    AllSamplesProfilingStopStrategy,
    NumStepsMeasuringStopStrategy,
    ProfilingConfig,
)
from aitune.torch.tune_strategy.mixin.performance_validation_mixin import (
    PerformanceValidationMixin,
    PerformanceValidationMixinConfig,
)
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy
from aitune.torch.tuning import tune

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


class TinyModel(torch.nn.Module):
    """Small model for performance validation tests."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


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


def _recording_sink(messages: list[str]):
    def sink(message: str, *args):
        messages.append(message % args if args else message)

    return sink


def _fast_profile_config() -> ProfilingConfig:
    return ProfilingConfig(
        batch_sizes=[1],
        measurement_stop_strategy=NumStepsMeasuringStopStrategy(num_steps=1),
        profiling_stop_strategy=AllSamplesProfilingStopStrategy(),
    )


def _strategy(messages: list[str]) -> OneBackendStrategy:
    strategy = OneBackendStrategy(
        TorchEagerBackend(),
        perf_validation_config=PerformanceValidationMixinConfig(profiling_config=_fast_profile_config()),
        sink=_recording_sink(messages),
    )
    strategy.enable_find_max_batch_size(False)
    return strategy


def _tune_tiny_model(strategy: OneBackendStrategy, torch_device):
    model = TinyModel().to(torch_device).eval()
    data = torch.randn(4, device=torch_device)

    with torch.no_grad():
        expected = model(data.unsqueeze(0))

    module = Module(model, "performance-validation-toggle", strategy=strategy)
    try:
        tune(
            module,
            data,
            batch_sizes=[1, 2],
            dry_run=False,
            device=torch_device,
            disable_external_logging=False,
            ignore_failing_modules=False,
        )
        actual = module(data.unsqueeze(0))
        torch.testing.assert_close(actual, expected)
    finally:
        MODULE_REGISTRY.clear()


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
    ext._resolved_batch_size = 99

    with (
        patch(
            "aitune.torch.tune_strategy.mixin.performance_validation_mixin.TorchEagerBackend",
            return_value=mock_eager_backend,
        ),
        patch(_PATCH_FIND_MAX_THROUGHPUT, return_value=(4, 100.0, MagicMock())),
    ):
        ext._pre_tune(mock_module, "mod", mock_graph_spec, mock_data, torch_device, tmp_path)

    assert ext.perf_validation_results == []
    assert ext._resolved_batch_size == mock_graph_spec.get_max_batch_size.return_value


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


def test_pre_tune_skips_baseline_when_performance_validation_disabled(
    mock_module, mock_graph_spec, mock_data, mock_eager_backend, torch_device, tmp_path
):
    """_pre_tune skips TorchEager baseline profiling when performance validation is disabled."""
    sink = MagicMock()
    ext = _ConcreteExtension(sink=sink)
    ext.enable_performance_validation(False)
    ext.enable_find_max_batch_size(False)
    ext._resolved_batch_size = 99

    with (
        patch(
            "aitune.torch.tune_strategy.mixin.performance_validation_mixin.TorchEagerBackend",
            return_value=mock_eager_backend,
        ) as mock_eager_cls,
        patch(_PATCH_FIND_MAX_THROUGHPUT) as mock_profile,
    ):
        ext._pre_tune(mock_module, "mod", mock_graph_spec, mock_data, torch_device, tmp_path)

    mock_eager_cls.assert_not_called()
    mock_profile.assert_not_called()
    mock_graph_spec.get_max_batch_size.assert_not_called()
    assert ext._baseline_throughput is None
    assert ext._baseline_backend is None
    assert ext._resolved_batch_size is None
    assert any("Performance validation against eager baseline is disabled" in str(call) for call in sink.call_args_list)


def test_performance_validation_enabled_by_default():
    """Performance validation is enabled by default for backwards-compatible behavior."""
    ext = _ConcreteExtension()

    assert ext._performance_validation_enabled is True
    assert not hasattr(ext, "validate_against_baseline")
    assert not hasattr(ext, "enable_validate_against_baseline")


def test_enable_performance_validation_returns_self_and_sets_flag():
    """enable_performance_validation toggles baseline profiling and performance checks."""
    ext = _ConcreteExtension()

    assert ext.enable_performance_validation() is ext
    assert ext._performance_validation_enabled is True

    assert ext.enable_performance_validation(False) is ext
    assert ext._performance_validation_enabled is False


# ── batch size resolution ─────────────────────────────────────────────────


def test_resolved_batch_size_uses_graph_spec_get_max_batch_size(
    mock_module, mock_graph_spec, mock_data, mock_eager_backend, torch_device, tmp_path
):
    """_resolved_batch_size is taken from graph_spec.get_max_batch_size()."""
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
    mock_graph_spec.get_max_batch_size.assert_called_once_with(normalized=True)


def test_pre_tune_profiles_baseline_when_validation_enabled(
    mock_module, mock_graph_spec, mock_data, mock_eager_backend, torch_device, tmp_path
):
    """_pre_tune profiles TorchEager baseline when validation is enabled."""
    ext = _ConcreteExtension()
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


def test_performance_validation_profiles_baseline_by_default(torch_device):
    messages: list[str] = []
    strategy = _strategy(messages)

    _tune_tiny_model(strategy, torch_device)

    assert strategy._baseline_throughput is not None
    assert strategy.perf_validation_results
    assert any("🔄 Profiling eager baseline...please wait" in message for message in messages)
    assert any("📊 Eager baseline: batch size=" in message for message in messages)


def test_performance_validation_can_skip_baseline_profiling(torch_device):
    messages: list[str] = []
    strategy = _strategy(messages)
    strategy.enable_performance_validation(False)

    _tune_tiny_model(strategy, torch_device)

    assert strategy._baseline_throughput is None
    assert strategy.perf_validation_results == []
    assert any("Performance validation against eager baseline is disabled" in message for message in messages)
    assert not any("🔄 Profiling eager baseline...please wait" in message for message in messages)


def test_pre_tune_logs_baseline_profiling_start_when_validation_enabled(
    mock_module, mock_graph_spec, mock_data, mock_eager_backend, torch_device, tmp_path
):
    """_pre_tune emits a visible message before profiling the TorchEager baseline."""
    sink = MagicMock()
    ext = _ConcreteExtension(sink=sink)
    ext.enable_find_max_batch_size(False)

    with (
        patch(
            "aitune.torch.tune_strategy.mixin.performance_validation_mixin.TorchEagerBackend",
            return_value=mock_eager_backend,
        ),
        patch(_PATCH_FIND_MAX_THROUGHPUT, return_value=(4, 80.0, MagicMock())),
    ):
        ext._pre_tune(mock_module, "mod", mock_graph_spec, mock_data, torch_device, tmp_path)

    assert any("🔄 Profiling eager baseline...please wait" in str(call) for call in sink.call_args_list)


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


def test_check_perf_skips_profiling_when_performance_validation_disabled(
    mock_module, mock_graph_spec, mock_data, mock_backend, torch_device, tmp_path
):
    """When performance validation is disabled, candidate performance checks are skipped."""
    ext = _ConcreteExtension()
    ext.enable_performance_validation(False)
    ext._baseline_throughput = 2.0
    ext._resolved_batch_size = 4

    with (
        patch.object(ext, "_build_and_validate_backend", return_value=mock_backend),
        patch(_PATCH_FIND_MAX_THROUGHPUT) as mock_profile,
    ):
        result = ext._build_validate_and_check_perf(
            mock_backend, mock_module, "mod", mock_graph_spec, mock_data, torch_device, tmp_path
        )

    assert result is mock_backend
    mock_profile.assert_not_called()
    assert ext.perf_validation_results == []


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
    assert config.min_speedup_ratio == 0.01
    assert config.profiling_config is None


def test_perf_validation_config_custom_threshold():
    config = PerformanceValidationMixinConfig(min_speedup_ratio=0.10)
    assert config.min_speedup_ratio == 0.10


@pytest.mark.parametrize("min_speedup_ratio", [-0.01, 1.01])
def test_perf_validation_config_rejects_invalid_min_speedup_ratio(min_speedup_ratio: float):
    with pytest.raises(ValueError, match="value must be between 0 and 1"):
        PerformanceValidationMixinConfig(min_speedup_ratio=min_speedup_ratio)


def test_profiling_config_for_batch_size_default_uses_num_steps():
    """Default profiling config uses NumStepsMeasuringStopStrategy with global defaults."""
    from aitune.torch.task.profiling import NumStepsMeasuringStopStrategy

    config = PerformanceValidationMixinConfig()
    profiling_cfg = config.profiling_config_for_batch_size(8)

    assert profiling_cfg.batch_sizes == [8]
    assert profiling_cfg.batching is True
    assert isinstance(profiling_cfg.measurement_stop_strategy, NumStepsMeasuringStopStrategy)
    assert profiling_cfg.measurement_stop_strategy.num_steps == 20
    assert profiling_cfg.measurement_stop_strategy.warmup_samples == 10


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


def test_post_tune_emits_baseline_selected_when_info_enabled(mock_graph_spec, mock_data):
    """When fallback selects TorchEager baseline, tuning explicitly reports it."""
    from unittest.mock import patch

    sink = MagicMock()
    ext = _ConcreteExtension(sink=sink)
    baseline = MagicMock(spec=Backend)
    baseline.describe.return_value = "TorchEagerBackend()"
    ext.perf_validation_results = []
    ext._baseline_backend = baseline
    ext._baseline_throughput = 123.0

    with patch.object(ext._logger, "isEnabledFor", return_value=True):
        ext._post_tune(baseline, "my_module", mock_graph_spec, mock_data)

    sink.assert_called_once()
    msg = sink.call_args[0][0]
    assert "Baseline was selected" in msg
    assert "TorchEagerBackend()" in msg
    assert "123.00 samples/s" in msg


def test_post_tune_silent_for_baseline_selected_when_info_disabled(mock_graph_spec, mock_data):
    """Baseline selection remains silent when INFO logging is disabled."""
    from unittest.mock import patch

    sink = MagicMock()
    ext = _ConcreteExtension(sink=sink)
    baseline = MagicMock(spec=Backend)
    baseline.describe.return_value = "TorchEagerBackend()"
    ext.perf_validation_results = []
    ext._baseline_backend = baseline
    ext._baseline_throughput = 123.0

    with (
        patch.object(ext._logger, "warning") as mock_warn,
        patch.object(ext._logger, "isEnabledFor", return_value=False),
    ):
        ext._post_tune(baseline, "my_module", mock_graph_spec, mock_data)

    sink.assert_not_called()
    mock_warn.assert_not_called()


def test_post_tune_silent_when_validation_disabled(mock_backend, mock_graph_spec, mock_data):
    """No performance summary is emitted when validation is disabled."""
    from unittest.mock import patch

    ext = _ConcreteExtension()
    ext.enable_performance_validation(False)
    ext.perf_validation_results = []

    with (
        patch.object(ext._logger, "isEnabledFor", return_value=False),
        patch.object(ext._logger, "warning") as mock_warn,
    ):
        ext._post_tune(mock_backend, "mod", mock_graph_spec, mock_data)

    mock_warn.assert_not_called()


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
