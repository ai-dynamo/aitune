# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from aitune.torch import Module
from aitune.torch.backend import Backend
from aitune.torch.backend.torch_eager import TorchEagerBackend
from aitune.torch.backend.torch_inductor_jit_backend import TorchInductorJitBackend
from aitune.torch.module.wrapper_module import ModuleState
from aitune.torch.task.correctness import CorrectnessValueError
from aitune.torch.task.profiling import (
    ModelExecutionTimeMeasuringStrategy,
    NumStepsMeasuringStopStrategy,
    ProfilingConfig,
)
from aitune.torch.task.profiling.profiling_stop_strategy import AllSamplesProfilingStopStrategy
from aitune.torch.tune_strategy.min_latency_strategy import MinLatencyProfilingResult, MinLatencyStrategy
from aitune.torch.tuning import tune
from tests.toy_backends import BuildFailsBackend, SleepBackend
from tests.toy_models.torch_models import ToyTorchModel
from tests.utilities.helpers import requires_cuda


@pytest.fixture
def mock_backend():
    """Create a mock backend for testing."""
    backend = MagicMock(spec=Backend)
    backend.name = "mock_backend"
    backend.describe.return_value = "mock_backend"
    return backend


def _profiling_config() -> ProfilingConfig:
    return ProfilingConfig(
        batch_sizes=[1],
        measuring_strategy=ModelExecutionTimeMeasuringStrategy(),
        measurement_stop_strategy=NumStepsMeasuringStopStrategy(num_steps=3),
        profiling_stop_strategy=AllSamplesProfilingStopStrategy(),
    )


def test_describe(mock_backend):
    """Test describe method."""
    strategy = MinLatencyStrategy(backends=[mock_backend, mock_backend], profiling_config=_profiling_config())
    strategy.enable_performance_validation(False)
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    description = strategy.describe()
    assert "name: Min Latency Strategy" in description
    assert "mock_backend" in description
    assert description.startswith("name: Min Latency Strategy")


def test_user_provided_torch_eager_treated_as_regular_backend():
    """When the user explicitly provides TorchEagerBackend it is kept as a user backend."""
    eager = TorchEagerBackend()
    strategy = MinLatencyStrategy([eager])
    assert len(strategy._backends) == 1
    assert strategy._backends[0] is eager


def test_performance_validation_toggle_returns_self_and_sets_flag(mock_backend):
    """Performance validation can be disabled with the common toggle."""
    strategy = MinLatencyStrategy([mock_backend])
    assert strategy.enable_performance_validation(False) is strategy
    assert strategy._performance_validation_enabled is False


def test_min_latency_strategy_selects_faster_backend(torch_device, tmp_path):
    """MinLatencyStrategy selects the backend with lower latency."""
    slower = SleepBackend(sleep_time=1e-2)
    faster = SleepBackend(sleep_time=1e-5)
    strategy = MinLatencyStrategy(backends=[slower, faster], profiling_config=_profiling_config())
    strategy.enable_performance_validation(False)
    strategy.enable_correctness_check(False)

    model = ToyTorchModel()
    sample = model.sample().unsqueeze(0)

    def mock_measure(backend, name, graph_spec, data, profiling_cfg):
        return MinLatencyProfilingResult(latency=backend.sleep_time)

    strategy._measure = mock_measure

    selected = strategy.tune(
        model, "test", model.graph_spec(batch_sizes=[1, 2]), [((sample,), {})], torch_device, tmp_path
    )

    assert isinstance(selected, SleepBackend)
    assert selected.sleep_time == faster.sleep_time


def test_min_latency_result_reports_batch_size_one():
    """MinLatencyProfilingResult exposes the fixed profiling batch size."""
    result = MinLatencyProfilingResult(latency=5.0)

    assert result.selected_batch_size == 1
    assert result.to_json_dict("latency") == {"latency": 5.0, "selected_batch_size": 1}


def test_min_latency_profiling_config_enforces_batch_size_one():
    """MinLatencyStrategy profiles with generated batch-size-1 inputs."""
    strategy = MinLatencyStrategy(backends=[], profiling_config=_profiling_config())

    profiling_config = strategy._get_profiling_config(batching=False, max_batch_size=8)

    assert profiling_config.batch_sizes == [1]
    assert profiling_config.batching is True


@requires_cuda
def test_min_latency_strategy_num_steps_all_samples(torch_device):
    strategy = MinLatencyStrategy(
        backends=[
            TorchInductorJitBackend(),
            TorchEagerBackend(),
        ],
        profiling_config=_profiling_config(),
    )
    strategy.enable_correctness_check(True)

    model = Module(ToyTorchModel().eval().to(torch_device), strategy=strategy)
    sample = model.sample().to(torch_device)
    batch_sizes = [1, 2, 4, 8, 16]
    n_backends = len(strategy._backends)

    tune(model, sample, batch_sizes=batch_sizes, device=torch_device, disable_external_logging=False)

    assert len(strategy.perf_validation_results) == n_backends
    assert all(r.metric > 0 for r in strategy.perf_validation_results)


def test_min_latency_strategy_fails_when_all_backends_fail_and_validation_disabled(torch_device):
    """If all backends fail and baseline validation is disabled, the module enters PASSTHROUGH."""
    strategy = MinLatencyStrategy(
        backends=[
            BuildFailsBackend(RuntimeError),
            BuildFailsBackend(MemoryError),
            BuildFailsBackend(CorrectnessValueError),
        ],
    )
    strategy.enable_performance_validation(False)
    model = ToyTorchModel().eval().to(torch_device)
    sample = model.sample().to(torch_device)

    model = Module(model, strategy=strategy)
    tune(model, sample, batch_sizes=list(range(1, 17)), device=torch_device, disable_external_logging=False)

    assert model.state == ModuleState.PASSTHROUGH


def test_min_latency_strategy_fallback_to_baseline_when_all_user_backends_fail(torch_device):
    """When all user backends fail and baseline validation is enabled, falls back to TorchEager."""
    strategy = MinLatencyStrategy(
        backends=[
            BuildFailsBackend(RuntimeError),
            BuildFailsBackend(MemoryError),
            BuildFailsBackend(CorrectnessValueError),
        ],
    )
    model = ToyTorchModel().eval().to(torch_device)
    sample = model.sample().to(torch_device)

    model = Module(model, strategy=strategy)
    tune(model, sample, batch_sizes=list(range(1, 17)), device=torch_device, disable_external_logging=False)

    assert model(sample.unsqueeze(0)) is not None


class _BuildFailsBackend(BuildFailsBackend):
    def __init__(self):
        super().__init__(RuntimeError)


def test_min_latency_strategy_find_max_batch_size_fails(torch_device):
    """If find max batch size fails, there is no recovery."""
    strategy = MinLatencyStrategy(backends=[SleepBackend()])
    strategy.set_find_max_batch_size_default_backend_class(_BuildFailsBackend)
    strategy.enable_find_max_batch_size(True)

    model = Module(ToyTorchModel().eval().to(torch_device), strategy=strategy)
    tune(
        model,
        model.__wrapped__.sample().to(torch_device),
        batch_sizes=list(range(1, 17)),
        device=torch_device,
        disable_external_logging=False,
    )

    assert model.state == ModuleState.PASSTHROUGH


def test_min_latency_perf_validation_results_populated(torch_device, tmp_path):
    """perf_validation_results is populated for user-provided backends only (not the baseline)."""
    slower = SleepBackend(sleep_time=1e-2)
    faster = SleepBackend(sleep_time=1e-5)
    strategy = MinLatencyStrategy(backends=[slower, faster], profiling_config=_profiling_config())
    strategy.enable_correctness_check(False)

    model = ToyTorchModel().eval().to(torch_device)
    sample = model.sample().unsqueeze(0).to(torch_device)

    strategy.tune(
        model,
        "test",
        model.graph_spec(batch_sizes=[1, 2], device=torch_device),
        [((sample,), {})],
        torch_device,
        tmp_path,
    )

    descriptions = [r.backend_description for r in strategy.perf_validation_results]
    assert not any("TorchEager" in d for d in descriptions)
    assert len(strategy.perf_validation_results) == 2
    for result in strategy.perf_validation_results:
        assert result.baseline_metric > 0
        assert result.metric > 0
        assert result.speedup > 0


def test_min_latency_post_tune_emits_speedup_summary(torch_device, tmp_path):
    """_post_tune emits a speedup line after successful tuning."""
    from unittest.mock import patch

    user_backend = SleepBackend(sleep_time=1e-5)
    strategy = MinLatencyStrategy(backends=[user_backend], profiling_config=_profiling_config())
    strategy._baseline_result = MinLatencyProfilingResult(latency=10.0)
    strategy._record_perf_result(user_backend, MinLatencyProfilingResult(latency=5.0))
    strategy.enable_correctness_check(False)

    with (
        patch.object(strategy._logger, "isEnabledFor", return_value=False),
        patch.object(strategy._logger, "warning") as mock_warn,
    ):
        strategy._post_tune(user_backend, "test", MagicMock(), [])

    mock_warn.assert_called()
    speedup_msgs = [c for c in mock_warn.call_args_list if "speedup:" in str(c).lower()]
    assert len(speedup_msgs) == 1
    assert "test" in str(speedup_msgs[0])
    assert "ms" in str(speedup_msgs[0])


def test_min_latency_post_tune_emits_baseline_selected_when_baseline_wins(mock_backend):
    """When TorchEager baseline wins, tuning explicitly reports baseline selection."""
    from unittest.mock import patch

    sink = MagicMock()
    strategy = MinLatencyStrategy(backends=[mock_backend], sink=sink)
    baseline = MagicMock(spec=Backend)
    baseline.describe.return_value = "TorchEagerBackend()"
    strategy._baseline_backend = baseline
    strategy._baseline_result = MinLatencyProfilingResult(latency=5.0)
    strategy.perf_validation_results = []

    with patch.object(strategy._logger, "isEnabledFor", return_value=True):
        strategy._post_tune(baseline, "test", MagicMock(), [])

    sink.assert_called_once()
    msg = sink.call_args[0][0]
    assert "Baseline was selected" in msg
    assert "TorchEagerBackend()" in msg
    assert "5.000 ms" in msg
