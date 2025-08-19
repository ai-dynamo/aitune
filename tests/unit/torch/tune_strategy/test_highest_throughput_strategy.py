# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from typing import cast
from unittest.mock import MagicMock

import pytest

from aitune.torch import Module
from aitune.torch.backend import Backend
from aitune.torch.backend.torch_eager import TorchEagerBackend
from aitune.torch.backend.torch_inductor_backend import TorchInductorBackend
from aitune.torch.module.wrapper_module import get_object_name
from aitune.torch.task.correctness import CorrectnessValueError
from aitune.torch.task.profiling import NumStepsMeasuringStopStrategy, StableWindowMeasuringStopStrategy
from aitune.torch.task.profiling.profiling_stop_strategy import (
    AllSamplesProfilingStopStrategy,
    ThroughputSaturatedProfilingStopStrategy,
)
from aitune.torch.tune_strategy.extension import TuneStrategyFindMaxBatchSizeExtension
from aitune.torch.tune_strategy.highest_throughput_strategy import HighestThroughputStrategy
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


def test_describe(mock_backend):
    """Test describe method."""
    backends = [mock_backend, mock_backend]
    strategy = HighestThroughputStrategy(backends)
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    assert (
        strategy.describe()
        == "name: Highest Throughput Strategy\ndescription: evaluate all backends, return backend with highest throughput\nbackends:\n  mock_backend\n  mock_backend"
    )


def test_sanity_batch_sizes_generator():
    max_batch_size = 2**20
    batch_sizes = [2**n for n in range(max_batch_size.bit_length() + 1)]
    assert 2**20 in batch_sizes


def test_highest_throughput_strategy_tune_highest_throughput_backend(torch_device, tmp_path):
    profiling_config = TuneStrategyFindMaxBatchSizeExtension.default_profiling_config(max_batch_size=8)
    slower = SleepBackend(sleep_time=1e-2)
    faster = SleepBackend(sleep_time=1e-5)
    strategy = HighestThroughputStrategy(backends=[slower, faster])
    strategy.set_find_max_batch_size_profiling_config(profiling_config)
    strategy.enable_correctness_check(False)

    model = ToyTorchModel()
    sample = model.sample().unsqueeze(0)  # as dataloader make batches, we need to unsqueeze the sample

    def forward_slow(x):
        time.sleep(1e-3)
        return x

    model.forward = forward_slow

    highest_throughput_backend = strategy.tune(
        model, "test", model.graph_spec(batch_sizes=[1, 2]), [((sample,), {})], torch_device, tmp_path
    )
    highest_throughput_backend = cast(SleepBackend, highest_throughput_backend)

    assert highest_throughput_backend.sleep_time == faster.sleep_time


def test_highest_throughput_strategy_max_batch_size_in_graph_spec(torch_device, tmp_path):
    profiling_config = TuneStrategyFindMaxBatchSizeExtension.default_profiling_config(max_batch_size=8)
    strategy = HighestThroughputStrategy(backends=[SleepBackend(sleep_time=1e-5)])
    strategy.set_find_max_batch_size_profiling_config(profiling_config)
    strategy.enable_correctness_check(False)
    model = ToyTorchModel().eval().to(torch_device)
    sample = model.sample().unsqueeze(0).to(torch_device)  # as dataloader make batches, we need to unsqueeze the sample
    graph_spec = model.graph_spec(batch_sizes=[1, 2, 4, 8], device=torch_device)

    # sanity heck that graph spec input_spec was updated with max batch size
    max_batch_sample = graph_spec.input_spec.make_batch(((sample,), {}), 8)
    assert max_batch_sample[0][0].shape[0] == 8

    # Run tuning to update graph spec with max batch size
    strategy.tune(model, "test", graph_spec, [((sample,), {})], torch_device, cache_dir=tmp_path)

    # Verify tensor specs were updated with max batch size info
    for tensor_spec in graph_spec.input_spec._tensor_specs:
        assert tensor_spec.max_shape[0] == 8


@requires_cuda
def test_highest_throughput_strategy_num_steps_all_samples(torch_device):
    find_profiling_config = TuneStrategyFindMaxBatchSizeExtension.default_profiling_config(max_batch_size=16)
    find_profiling_config.measurement_stop_strategy = NumStepsMeasuringStopStrategy(num_steps=10)

    strategy = HighestThroughputStrategy(
        backends=[
            TorchInductorBackend(),
            TorchEagerBackend(),
        ],
        measurement_stop_strategy=NumStepsMeasuringStopStrategy(num_steps=10),
        profiling_stop_strategy=AllSamplesProfilingStopStrategy(),
    )
    strategy.set_find_max_batch_size_profiling_config(find_profiling_config)
    strategy.enable_correctness_check(True)

    model = Module(ToyTorchModel().eval().to(torch_device), strategy=strategy)
    sample = model.sample().to(torch_device)
    batch_sizes = [1, 2, 4, 8, 16]
    n_backends = len(strategy._backends)
    n_batch_sizes = len(batch_sizes)
    n_steps = find_profiling_config.measurement_stop_strategy.num_steps

    tune(model, sample, batch_sizes=batch_sizes, device=torch_device, disable_external_logging=False)

    assert len(strategy.results) == 1  # for 1 graph spec

    assert strategy.results[0].graph_spec_name == "0"

    assert len(strategy.results[0].measurements) == n_steps * n_batch_sizes * n_backends
    assert len(strategy.results[0].highest_throughput_results) == n_backends

    assert all(m.model_name == get_object_name(model) for m in strategy.results[0].measurements)
    assert all(m.backend_details is not None for m in strategy.results[0].measurements)
    assert all(m.execution_time > 0 for m in strategy.results[0].measurements)

    # all measurement ids are unique
    assert len({m.measurement_id for m in strategy.results[0].measurements}) == len(strategy.results[0].measurements)

    assert len({m.batch_size for m in strategy.results[0].measurements}) == n_batch_sizes

    # check graph spec
    graph_specs = list(model._self_wrapper._backends.keys())
    assert len(graph_specs) == 1
    assert graph_specs[0].tensor_specs[0].max_shape[0] == 16


@requires_cuda
def test_highest_throughput_strategy_stable_window(torch_device):
    strategy = HighestThroughputStrategy(
        backends=[
            TorchInductorBackend(),
        ],
        measurement_stop_strategy=StableWindowMeasuringStopStrategy(window_size=10, stability_percentage=90),
        profiling_stop_strategy=ThroughputSaturatedProfilingStopStrategy(throughput_cutoff_threshold=0.99),
    ).enable_find_max_batch_size(False)
    model = Module(ToyTorchModel().eval().to(torch_device), strategy=strategy)
    sample = model.sample().to(torch_device)

    batch_sizes = list(range(1, 17))
    n_backends = len(strategy._backends)
    n_steps = 10

    tune(model, sample, batch_sizes=batch_sizes, device=torch_device, disable_external_logging=False)

    assert len(strategy.results) == 1  # for 1 graph spec

    assert strategy.results[0].graph_spec_name == "0"

    assert len(strategy.results[0].measurements) >= n_backends * n_steps
    assert len(strategy.results[0].highest_throughput_results) == n_backends

    assert all(m.model_name == get_object_name(model) for m in strategy.results[0].measurements)
    assert all(m.backend_details is not None for m in strategy.results[0].measurements)
    assert all(m.execution_time > 0 for m in strategy.results[0].measurements)

    # all measurement ids are unique
    assert len({m.measurement_id for m in strategy.results[0].measurements}) == len(strategy.results[0].measurements)

    assert len({m.batch_size for m in strategy.results[0].measurements}) >= 1


class ActivateFailsBackend(SleepBackend):
    def __init__(self):
        super().__init__()

    def _activate(self):
        raise RuntimeError("Activate failed")


def test_highest_throughput_strategy_fails_backend_if_all_of_backends_fails(torch_device):
    """If all backends fails it should raise an error."""
    strategy = HighestThroughputStrategy(
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
    with pytest.raises(RuntimeError, match="No correct backend found with throughput > 0"):
        tune(model, sample, batch_sizes=batch_sizes, device=torch_device, disable_external_logging=False)


def test_highest_throughput_strategy_select_backend_if_one_of_backends_succeeds(torch_device):
    """If backend fails it should be skipped and TorchEagerBackend should be used as a fallback."""
    strategy = HighestThroughputStrategy(
        backends=[
            BuildFailsBackend(RuntimeError),
            BuildFailsBackend(MemoryError),
            BuildFailsBackend(CorrectnessValueError),
            ActivateFailsBackend(),
            TorchEagerBackend(),
        ],
    )
    model = ToyTorchModel().eval().to(torch_device)
    sample = model.sample().to(torch_device)
    assert model(sample) is not None

    batch_sizes = list(range(1, 17))

    model = Module(model, strategy=strategy)
    tune(model, sample, batch_sizes=batch_sizes, device=torch_device, disable_external_logging=False)

    assert model(sample.unsqueeze(0)) is not None  # Note: does not work without unsqueeze


def test_highest_throughput_strategy_find_max_batch_size_fails(torch_device):
    """If find max batch size fails, there is no recovery."""
    strategy = HighestThroughputStrategy(
        backends=[
            SleepBackend(),
        ],
    )

    # failing backend
    strategy.find_config.default_backend_class = lambda: BuildFailsBackend(RuntimeError)
    strategy.enable_find_max_batch_size(True)

    model = ToyTorchModel().eval().to(torch_device)
    sample = model.sample().to(torch_device)
    assert model(sample) is not None

    batch_sizes = list(range(1, 17))

    model = Module(model, strategy=strategy)

    with pytest.raises(RuntimeError, match="Build failed"):
        tune(model, sample, batch_sizes=batch_sizes, device=torch_device, disable_external_logging=False)
