# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast
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
    StableWindowMeasuringStopStrategy,
    ThroughputSaturatedProfilingStopStrategy,
)
from aitune.torch.task.profiling.profiling_stop_strategy import AllSamplesProfilingStopStrategy
from aitune.torch.tune_strategy.max_throughput_strategy import MaxThroughputProfilingResult, MaxThroughputStrategy
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


def _profiling_config(
    max_batch_size: int = 8,
    measurement_stop_strategy=None,
    profiling_stop_strategy=None,
) -> ProfilingConfig:
    return ProfilingConfig(
        batch_sizes=[2**n for n in range(max_batch_size.bit_length())],
        measuring_strategy=ModelExecutionTimeMeasuringStrategy(),
        measurement_stop_strategy=measurement_stop_strategy or NumStepsMeasuringStopStrategy(num_steps=3),
        profiling_stop_strategy=profiling_stop_strategy or AllSamplesProfilingStopStrategy(),
    )


def test_describe(mock_backend):
    """Test describe method."""
    backends = [mock_backend, mock_backend]
    strategy = MaxThroughputStrategy(backends)
    strategy.enable_performance_validation(False)
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    description = strategy.describe()
    assert "name: Max Throughput Strategy" in description
    assert "mock_backend" in description
    assert description.startswith("name: Max Throughput Strategy")


def test_user_provided_torch_eager_treated_as_regular_backend():
    """When the user explicitly provides TorchEagerBackend it is kept as a user backend."""
    eager = TorchEagerBackend()
    strategy = MaxThroughputStrategy([eager])
    assert len(strategy._backends) == 1
    assert strategy._backends[0] is eager


def test_torch_eager_not_injected_when_already_present():
    """TorchEager is not duplicated when already in the provided backends list."""
    eager = TorchEagerBackend()
    strategy = MaxThroughputStrategy([eager])
    assert len(strategy._backends) == 1
    assert strategy._backends[0] is eager


def test_performance_validation_toggle_returns_self_and_sets_flag(mock_backend):
    """Performance validation can be disabled with the common toggle."""
    strategy = MaxThroughputStrategy([mock_backend])
    assert strategy.enable_performance_validation(False) is strategy
    assert strategy._performance_validation_enabled is False
    assert len(strategy._backends) == 1
    assert not any(isinstance(b, TorchEagerBackend) for b in strategy._backends)


def test_sanity_batch_sizes_generator():
    max_batch_size = 2**20
    batch_sizes = [2**n for n in range(max_batch_size.bit_length() + 1)]
    assert 2**20 in batch_sizes


def test_max_throughput_uses_strategy_profiling_config():
    """MaxThroughputStrategy derives sweep configs from one strategy-level profiling config."""
    measurement_stop_strategy = NumStepsMeasuringStopStrategy(num_steps=3)
    profiling_stop_strategy = AllSamplesProfilingStopStrategy()
    profiling_config = _profiling_config(
        max_batch_size=8,
        measurement_stop_strategy=measurement_stop_strategy,
        profiling_stop_strategy=profiling_stop_strategy,
    )

    backend = MagicMock(spec=Backend)
    strategy = MaxThroughputStrategy(backends=[backend], profiling_config=profiling_config)
    result = strategy._get_profiling_config(batching=True, max_batch_size=8)

    assert result.batch_sizes == [1, 2, 4, 8]
    assert result.batching is True
    assert result.measurement_stop_strategy is measurement_stop_strategy
    assert result.profiling_stop_strategy is profiling_stop_strategy


def test_max_throughput_strategy_tune_max_throughput_backend(torch_device, tmp_path):
    slower = SleepBackend(sleep_time=1e-2)
    faster = SleepBackend(sleep_time=1e-5)
    strategy = MaxThroughputStrategy(
        backends=[slower, faster],
        profiling_config=_profiling_config(),
    )
    strategy.enable_performance_validation(False)
    strategy.enable_correctness_check(False)

    model = ToyTorchModel().to(torch_device)
    sample = model.sample().unsqueeze(0).to(torch_device)  # as dataloader make batches, we need to unsqueeze the sample

    def mock_measure(backend, name, graph_spec, data, profiling_cfg):
        return MaxThroughputProfilingResult(throughput=1.0 / backend.sleep_time, selected_batch_size=1)

    strategy._measure = mock_measure

    max_throughput_backend = strategy.tune(
        model,
        "test",
        model.graph_spec(batch_sizes=[1, 2], device=torch_device),
        [((sample,), {})],
        torch_device,
        tmp_path,
    )
    max_throughput_backend = cast(SleepBackend, max_throughput_backend)

    assert max_throughput_backend.sleep_time == faster.sleep_time


def test_max_throughput_strategy_max_batch_size_in_graph_spec(torch_device, tmp_path):
    strategy = MaxThroughputStrategy(
        backends=[SleepBackend(sleep_time=1e-5)],
        profiling_config=_profiling_config(),
    )
    strategy.enable_correctness_check(False)
    model = ToyTorchModel().eval().to(torch_device)
    sample = model.sample().unsqueeze(0).to(torch_device)  # as dataloader make batches, we need to unsqueeze the sample
    graph_spec = model.graph_spec(batch_sizes=[1, 2, 4, 8], device=torch_device)

    # sanity heck that graph spec input_spec was updated with max batch size
    max_batch_args, _ = graph_spec.make_batch((sample,), {}, batch_size=8)
    assert max_batch_args[0].shape[0] == 8

    # Run tuning to update graph spec with max batch size
    strategy.tune(model, "test", graph_spec, [((sample,), {})], torch_device, cache_dir=tmp_path)

    # Verify tensor specs were updated with max batch size info
    for tensor_spec in graph_spec.input_spec.tensor_specs:
        assert tensor_spec.max_shape[0] == 8


@requires_cuda
def test_max_throughput_strategy_num_steps_all_samples(torch_device):
    strategy = MaxThroughputStrategy(
        backends=[
            TorchInductorJitBackend(),
            TorchEagerBackend(),
        ],
        profiling_config=_profiling_config(
            max_batch_size=16,
            measurement_stop_strategy=NumStepsMeasuringStopStrategy(num_steps=10),
            profiling_stop_strategy=AllSamplesProfilingStopStrategy(),
        ),
    )
    strategy.enable_correctness_check(True)

    model = Module(ToyTorchModel().eval().to(torch_device), strategy=strategy)
    sample = model.sample().to(torch_device)
    batch_sizes = [1, 2, 4, 8, 16]
    n_backends = len(strategy._backends)

    tune(model, sample, batch_sizes=batch_sizes, device=torch_device, disable_external_logging=False)

    assert len(strategy.perf_validation_results) == n_backends
    assert all(r.metric > 0 for r in strategy.perf_validation_results)

    # check graph spec
    graph_specs = list(model._self_wrapper._backends.keys())
    assert len(graph_specs) == 1
    assert graph_specs[0].tensor_specs[0].max_shape[0] == 16


@requires_cuda
def test_max_throughput_strategy_stable_window(torch_device):
    strategy = MaxThroughputStrategy(
        backends=[
            TorchInductorJitBackend(),
        ],
        profiling_config=_profiling_config(
            measurement_stop_strategy=StableWindowMeasuringStopStrategy(
                window_size=10,
                max_cv_ratio=0.90,
            ),
            profiling_stop_strategy=ThroughputSaturatedProfilingStopStrategy(
                min_throughput_gain_ratio=0.99,
                throughput_backoff_limit=0,
            ),
        ),
    ).enable_find_max_batch_size(False)
    model = Module(ToyTorchModel().eval().to(torch_device), strategy=strategy)
    sample = model.sample().to(torch_device)

    batch_sizes = list(range(1, 17))
    n_backends = len(strategy._backends)

    tune(model, sample, batch_sizes=batch_sizes, device=torch_device, disable_external_logging=False)

    assert len(strategy.perf_validation_results) == n_backends
    assert all(r.metric > 0 for r in strategy.perf_validation_results)


class ActivateFailsBackend(SleepBackend):
    def __init__(self):
        super().__init__()

    def _activate(self):
        raise RuntimeError("Activate failed")


class RuntimeErrorBuildFailsBackend(BuildFailsBackend):
    def __init__(self):
        super().__init__(RuntimeError)


def test_max_throughput_strategy_fails_backend_if_all_of_backends_fails(torch_device):
    """If all backends fails it should raise an error."""
    strategy = MaxThroughputStrategy(
        backends=[
            BuildFailsBackend(RuntimeError),
            BuildFailsBackend(MemoryError),
            BuildFailsBackend(CorrectnessValueError),
            ActivateFailsBackend(),
        ],
    )
    strategy.enable_performance_validation(False)
    model = ToyTorchModel().eval().to(torch_device)
    sample = model.sample().to(torch_device)
    assert model(sample) is not None

    batch_sizes = list(range(1, 17))

    model = Module(model, strategy=strategy)
    tune(model, sample, batch_sizes=batch_sizes, device=torch_device, disable_external_logging=False)

    assert model.state == ModuleState.PASSTHROUGH


def test_max_throughput_strategy_fallback_to_baseline_when_all_user_backends_fail(torch_device):
    """When all user backends fail and validation is enabled, falls back to TorchEager baseline."""
    strategy = MaxThroughputStrategy(
        backends=[
            BuildFailsBackend(RuntimeError),
            BuildFailsBackend(MemoryError),
            BuildFailsBackend(CorrectnessValueError),
            ActivateFailsBackend(),
        ],
    )
    model = ToyTorchModel().eval().to(torch_device)
    sample = model.sample().to(torch_device)
    assert model(sample) is not None

    batch_sizes = list(range(1, 17))

    model = Module(model, strategy=strategy)
    tune(model, sample, batch_sizes=batch_sizes, device=torch_device, disable_external_logging=False)

    assert model(sample.unsqueeze(0)) is not None  # Note: does not work without unsqueeze


def test_max_throughput_strategy_find_max_batch_size_fails(torch_device):
    """If find max batch size fails, there is no recovery."""
    strategy = MaxThroughputStrategy(
        backends=[
            SleepBackend(),
        ],
    )

    strategy.set_find_max_batch_size_default_backend_class(RuntimeErrorBuildFailsBackend)
    strategy.enable_find_max_batch_size(True)

    model = ToyTorchModel().eval().to(torch_device)
    sample = model.sample().to(torch_device)
    assert model(sample) is not None

    batch_sizes = list(range(1, 17))

    model = Module(model, strategy=strategy)

    tune(model, sample, batch_sizes=batch_sizes, device=torch_device, disable_external_logging=False)
    assert model.state == ModuleState.PASSTHROUGH


def test_max_throughput_perf_validation_results_populated(torch_device, tmp_path):
    """perf_validation_results is populated for user-provided backends only (not the baseline)."""
    slower = SleepBackend(sleep_time=1e-2)
    faster = SleepBackend(sleep_time=1e-5)
    strategy = MaxThroughputStrategy(
        backends=[slower, faster],
        profiling_config=_profiling_config(),
    )
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

    # TorchEager is the baseline (profiled in _pre_tune), not in perf_validation_results.
    # Only the two user-provided backends appear.
    descriptions = [r.backend_description for r in strategy.perf_validation_results]
    assert not any("TorchEager" in d for d in descriptions)
    assert len(strategy.perf_validation_results) == 2
    for result in strategy.perf_validation_results:
        assert result.baseline_metric > 0
        assert result.metric > 0
        assert result.speedup > 0


def test_max_throughput_torcheager_excluded_from_selection_when_performance_validation_disabled(torch_device, tmp_path):
    """When performance validation is disabled, TorchEager is never selected as an implicit baseline."""
    user_backend = SleepBackend(sleep_time=1e-5)
    strategy = MaxThroughputStrategy(
        backends=[user_backend],
        profiling_config=_profiling_config(),
    )
    strategy.enable_performance_validation(False)
    strategy.enable_correctness_check(False)

    model = ToyTorchModel().eval().to(torch_device)
    sample = model.sample().unsqueeze(0).to(torch_device)

    selected = strategy.tune(
        model,
        "test",
        model.graph_spec(batch_sizes=[1, 2], device=torch_device),
        [((sample,), {})],
        torch_device,
        tmp_path,
    )

    assert not isinstance(selected, TorchEagerBackend)
    assert strategy._baseline_backend is None
    assert strategy.perf_validation_results == []


def test_max_throughput_post_tune_emits_speedup_summary(torch_device, tmp_path):
    """_post_tune emits a speedup line after successful tuning."""
    from unittest.mock import patch

    user_backend = SleepBackend(sleep_time=1e-5)
    strategy = MaxThroughputStrategy(
        backends=[user_backend],
        profiling_config=_profiling_config(),
    )
    strategy._baseline_result = MaxThroughputProfilingResult(throughput=1.0, selected_batch_size=1)
    strategy._record_perf_result(user_backend, MaxThroughputProfilingResult(throughput=2.0, selected_batch_size=1))
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
    assert "samples/s" in str(speedup_msgs[0])


def test_max_throughput_post_tune_emits_baseline_selected_when_baseline_wins(mock_backend):
    """When TorchEager baseline wins, tuning explicitly reports baseline selection."""
    from unittest.mock import patch

    sink = MagicMock()
    strategy = MaxThroughputStrategy(backends=[mock_backend], sink=sink)
    baseline = MagicMock(spec=Backend)
    baseline.describe.return_value = "TorchEagerBackend()"
    strategy._baseline_backend = baseline
    strategy._baseline_result = MaxThroughputProfilingResult(throughput=42.0, selected_batch_size=1)
    strategy.perf_validation_results = []

    with patch.object(strategy._logger, "isEnabledFor", return_value=True):
        strategy._post_tune(baseline, "test", MagicMock(), [])

    sink.assert_called_once()
    msg = sink.call_args[0][0]
    assert "Baseline was selected" in msg
    assert "TorchEagerBackend()" in msg
    assert "42.00 samples/s" in msg


def test_max_throughput_post_tune_silent_for_baseline_selected_when_info_disabled(mock_backend):
    """Baseline selection remains silent when INFO logging is disabled."""
    from unittest.mock import patch

    sink = MagicMock()
    strategy = MaxThroughputStrategy(backends=[mock_backend], sink=sink)
    baseline = MagicMock(spec=Backend)
    baseline.describe.return_value = "TorchEagerBackend()"
    strategy._baseline_backend = baseline
    strategy._baseline_result = MaxThroughputProfilingResult(throughput=42.0, selected_batch_size=1)
    strategy.perf_validation_results = []

    with (
        patch.object(strategy._logger, "isEnabledFor", return_value=False),
        patch.object(strategy._logger, "warning") as mock_warn,
    ):
        strategy._post_tune(baseline, "test", MagicMock(), [])

    sink.assert_not_called()
    mock_warn.assert_not_called()
