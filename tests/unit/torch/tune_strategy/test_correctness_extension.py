# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from aitune.torch.backend.torch_eager import TorchEagerBackend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.task.correctness import (
    CorrectnessDynamicShapeError,
    CorrectnessTensorShapeError,
    CorrectnessValueError,
)
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

    def to_json_dict(self) -> dict[str, Any]:
        return {}


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


def test_correctness_requires_at_least_one_sample():
    """Correctness validation should not pass without recorded samples."""

    class MockBackend:
        def describe(self):
            return "mock_backend"

    strategy = TuneStrategyTestCorrectness()
    input_spec = output_spec = SampleMetadata.from_inputs((), {})
    graph_spec = GraphSpec(name="test_model", input_spec=input_spec, output_spec=output_spec)

    with pytest.raises(ValueError, match="requires at least one sample"):
        strategy.check_correctness(MockBackend(), "test_model", graph_spec, [])  # type: ignore


def test_correctness_checks_dynamic_min_and_max_shapes():
    """Dynamic graph correctness should validate recorded, min, and max input shapes."""

    class MockBackend:
        def __init__(self):
            self.input_shapes = []

        def infer(self, *args, **kwargs):
            del kwargs
            self.input_shapes.append(tuple(args[0].shape))
            return args[0]

        def describe(self):
            return "mock_backend"

    strategy = TuneStrategyTestCorrectness()
    backend = MockBackend()

    recorded_sample = ((torch.randn(2, 8),), {})
    min_sample = ((torch.randn(2, 4),), {})
    max_sample = ((torch.randn(2, 12),), {})
    input_spec = SampleMetadata.from_inputs(*recorded_sample, batch_size=2)
    input_spec.update_shapes_seen(SampleMetadata.from_inputs(*min_sample, batch_size=2))
    input_spec.update_shapes_seen(SampleMetadata.from_inputs(*max_sample, batch_size=2))
    output_spec = SampleMetadata.from_outputs(torch.randn(2, 8), batch_size=2)
    output_spec.update_shapes_seen(SampleMetadata.from_outputs(torch.randn(2, 4), batch_size=2))
    output_spec.update_shapes_seen(SampleMetadata.from_outputs(torch.randn(2, 12), batch_size=2))
    graph_spec = GraphSpec(name="test_model", input_spec=input_spec, output_spec=output_spec)

    strategy.check_correctness(backend, "test_model", graph_spec, [recorded_sample])  # type: ignore

    assert backend.input_shapes == [(2, 8), (2, 4), (2, 12)]


def test_correctness_skips_value_checks_for_dynamic_boundary_samples():
    """Dynamic boundary samples should only prove the backend executes those shapes."""

    class MockBackend:
        def __init__(self):
            self.input_shapes = []

        def infer(self, *args, **kwargs):
            del kwargs
            input_shape = tuple(args[0].shape)
            self.input_shapes.append(input_shape)
            if input_shape == (2, 8):
                return torch.zeros(2, 8)
            return torch.full((2, 8), float("nan"))

        def describe(self):
            return "mock_backend"

    strategy = TuneStrategyTestCorrectness()
    backend = MockBackend()

    recorded_sample = ((torch.randn(2, 8),), {})
    min_sample = ((torch.randn(2, 4),), {})
    max_sample = ((torch.randn(2, 12),), {})
    input_spec = SampleMetadata.from_inputs(*recorded_sample, batch_size=2)
    input_spec.update_shapes_seen(SampleMetadata.from_inputs(*min_sample, batch_size=2))
    input_spec.update_shapes_seen(SampleMetadata.from_inputs(*max_sample, batch_size=2))
    output_spec = SampleMetadata.from_outputs(torch.zeros(2, 8), batch_size=2)
    graph_spec = GraphSpec(name="test_model", input_spec=input_spec, output_spec=output_spec)

    strategy.check_correctness(backend, "test_model", graph_spec, [recorded_sample])  # type: ignore

    assert backend.input_shapes == [(2, 8), (2, 4), (2, 12)]


def test_correctness_reports_dynamic_boundary_inference_failure():
    """Dynamic boundary failures should be reported by the correctness layer."""

    class MockBackend:
        def infer(self, *args, **kwargs):
            del kwargs
            if tuple(args[0].shape) != (2, 8):
                raise RuntimeError("backend shape failure")
            return torch.zeros(2, 8)

        def describe(self):
            return "mock_backend"

    strategy = TuneStrategyTestCorrectness()

    recorded_sample = ((torch.randn(2, 8),), {})
    min_sample = ((torch.randn(2, 4),), {})
    max_sample = ((torch.randn(2, 12),), {})
    input_spec = SampleMetadata.from_inputs(*recorded_sample, batch_size=2)
    input_spec.update_shapes_seen(SampleMetadata.from_inputs(*min_sample, batch_size=2))
    input_spec.update_shapes_seen(SampleMetadata.from_inputs(*max_sample, batch_size=2))
    output_spec = SampleMetadata.from_outputs(torch.zeros(2, 8), batch_size=2)
    graph_spec = GraphSpec(name="test_model", input_spec=input_spec, output_spec=output_spec)

    with pytest.raises(
        CorrectnessDynamicShapeError,
        match="Dynamic shape correctness check failed.*min.*test_model.*mock_backend",
    ):
        strategy.check_correctness(MockBackend(), "test_model", graph_spec, [recorded_sample])  # type: ignore


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


def test_correctness_failure_is_appended_to_build_log(mocker, torch_device, tmp_path):
    """Validation failures after build should be written to the backend build log."""
    backend = mocker.MagicMock()
    backend.describe.return_value = "mock_backend"
    backend.key.return_value = "mock_backend_key"
    backend.__deepcopy__ = lambda _, memo=None: backend
    backend.build.return_value = backend
    backend.is_active = False
    module = mocker.MagicMock()
    strategy = TuneStrategyTestCorrectness()
    strategy.backend_results = []
    mocker.patch.object(strategy, "check_correctness", side_effect=RuntimeError("correctness failed"))
    sample = ((torch.randn(2, 8),), {})
    graph_spec = GraphSpec(
        name="test_model",
        input_spec=SampleMetadata.from_inputs(*sample, batch_size=2),
        output_spec=SampleMetadata.from_outputs(torch.randn(2, 8), batch_size=2),
    )

    result = strategy._build_and_validate_backend(
        backend,
        module,
        "test_model",
        graph_spec,
        [sample],
        torch_device,
        tmp_path,
    )

    assert result is None
    log_text = (tmp_path / "mock_backend_key" / "build.log").read_text(encoding="utf-8")
    assert "Backend build or validation failed" in log_text
    assert "Exception type: RuntimeError" in log_text
    assert "Exception details: correctness failed" in log_text
