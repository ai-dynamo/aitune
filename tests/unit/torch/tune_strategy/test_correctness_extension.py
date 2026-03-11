# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from aitune.torch.backend.torch_eager import TorchEagerBackend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.task.correctness import CorrectnessTensorShapeError, CorrectnessValueError
from aitune.torch.tune_strategy import TuneStrategy
from tests.toy_models.torch_models import ToyTorchModel


class TuneStrategyTestCorrectness(TuneStrategy):
    def _tune(
        self,
        module: nn.Module,
        name: str,
        graph_spec: GraphSpec,
        data: list[Sample],
        device: torch.device,
        cache_dir: Path,
    ):
        backend = TorchEagerBackend()
        backend = backend.build(module, graph_spec, data, device, cache_dir)
        self.check_correctness(backend, name, graph_spec, data)
        return backend

    def _describe_parts(self) -> list[str]:
        return ["TuneStrategyTestCorrectness"]


def test_correctness_extension_torch_eager_backend(torch_device, tmp_path):
    """Test correctness extension with torch eager backend."""

    module = ToyTorchModel()
    graph_spec = module.graph_spec()
    data = module.samples(device=torch_device)

    strategy = TuneStrategyTestCorrectness()

    backend = strategy.tune(module, "test_model", graph_spec, data, torch_device, cache_dir=tmp_path)
    backend.deactivate()


def test_correctness_is_idempotent():
    """Test correctness is idempotent, has no side effects."""

    class MockBackend:
        def infer(self, *args, **kwargs):
            """This mock backend alters cache in args and kwargs."""
            args[0].append("not important, should be discarded")
            kwargs["cache"].append("not important, should be discarded")
            return args, kwargs

        def describe(self):
            return "mock_backend"

    strategy = TuneStrategyTestCorrectness()

    cache = []
    data = [((cache,), {"cache": cache})]
    input_spec = output_spec = SampleMetadata.from_inputs((), {})
    graph_spec = GraphSpec(name="test_model", input_spec=input_spec, output_spec=output_spec)
    strategy.check_correctness(MockBackend(), "test_model", graph_spec, data)  # type: ignore
    assert len(cache) == 0, "cache should be empty"


def test_correctness_extension_torch_eager_backend_with_nan(torch_device, tmp_path):
    """Test correctness extension with torch eager backend."""

    module = ToyTorchModel()
    graph_spec = module.graph_spec()
    data = module.samples(device=torch_device)
    data[0][0][0][0] = float("nan")

    strategy = TuneStrategyTestCorrectness()

    with pytest.raises(CorrectnessValueError, match="contains NaN values"):
        backend = strategy.tune(module, "test_model", graph_spec, data, torch_device, cache_dir=tmp_path)
        backend.deactivate()


def test_correctness_extension_torch_eager_backend_with_inf(mocker, torch_device, tmp_path):
    """Test correctness extension with torch eager backend."""
    module = ToyTorchModel()
    graph_spec = module.graph_spec()
    data = module.samples(device=torch_device)
    data[0][0][0][0] = float("inf")

    mocker.patch.object(module, "forward", return_value=torch.tensor([float("inf")]))

    strategy = TuneStrategyTestCorrectness()

    with pytest.raises(CorrectnessValueError, match="contains infinity values"):
        backend = strategy.tune(module, "test_model", graph_spec, data, torch_device, cache_dir=tmp_path)
        backend.deactivate()


def test_correctness_extension_torch_eager_backend_with_wrong_shapes(torch_device, tmp_path):
    """Test correctness extension with torch eager backend."""
    module = ToyTorchModel()
    graph_spec = module.graph_spec()
    data = module.samples(device=torch_device, batch_sizes=[1])

    strategy = TuneStrategyTestCorrectness()

    with pytest.raises(
        CorrectnessTensorShapeError, match=r"Expected tensor outputs to have shape \[2, 5\] but got \[1, 5\]"
    ):
        backend = strategy.tune(module, "test_model", graph_spec, data, torch_device, cache_dir=tmp_path)
        backend.deactivate()
