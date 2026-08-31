# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from aitune.torch.backend.torch_eager import TorchEagerBackend
from aitune.torch.module.forward_signature import ForwardSignature
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.module.sample_store import Sample, SampleStore
from aitune.torch.task.correctness import (
    CorrectnessDynamicShapeError,
    CorrectnessTensorShapeError,
    CorrectnessValueError,
)
from aitune.torch.tune_strategy import TuneStrategy
from tests.toy_models.torch_models import ToyTorchModel
from tests.utilities.helpers import make_input_metadata, make_sample_store


def _forward(x=None, *, cache=None):
    return x, cache


FORWARD_SIGNATURE = ForwardSignature.from_callable(_forward)


def _input_metadata(sample: Sample, batch_size: int | None = None) -> SampleMetadata:
    return make_input_metadata(FORWARD_SIGNATURE, sample, batch_size=batch_size)


class TuneStrategyTestCorrectness(TuneStrategy):
    def _tune(
        self,
        module: nn.Module,
        name: str,
        graph_spec: GraphSpec,
        samples: SampleStore,
        device: torch.device,
        cache_dir: Path,
    ):
        backend = TorchEagerBackend()
        backend = backend.build(module, graph_spec, samples, device, cache_dir)
        self.check_correctness(backend, name, graph_spec, samples)
        return backend

    def _describe_parts(self) -> list[str]:
        return ["TuneStrategyTestCorrectness"]

    def to_json_dict(self) -> dict[str, Any]:
        return {}


def test_correctness_extension_torch_eager_backend(torch_device, tmp_path):
    """Test correctness extension with torch eager backend."""
    module = ToyTorchModel()
    graph_spec = module.graph_spec()
    samples = module.sample_store(tmp_path, device=torch_device)

    strategy = TuneStrategyTestCorrectness()

    backend = strategy.tune(module, "test_model", graph_spec, samples, torch_device, cache_dir=tmp_path)
    backend.deactivate()


def test_correctness_is_idempotent(tmp_path):
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
    samples = make_sample_store([((cache,), {"cache": cache})], tmp_path)
    input_spec = output_spec = _input_metadata(((), {}))
    graph_spec = GraphSpec(
        name="test_model",
        input_spec=input_spec,
        output_spec=output_spec,
        forward_signature=FORWARD_SIGNATURE,
    )
    strategy.check_correctness(MockBackend(), "test_model", graph_spec, samples)  # type: ignore
    assert len(cache) == 0, "cache should be empty"


def test_correctness_loads_samples_on_backend_device(mocker, tmp_path):
    """Correctness should remap recorded CUDA samples to the backend device."""

    class MockBackend:
        device = torch.device("cuda:1")

        def infer(self, value):
            return value

        def describe(self):
            return "mock_backend"

    sample = ((torch.zeros(2, 4),), {})
    samples = make_sample_store([sample], tmp_path)
    iter_samples = mocker.spy(samples, "iter_samples")
    graph_spec = GraphSpec(
        name="test_model",
        input_spec=_input_metadata(sample, batch_size=2),
        output_spec=SampleMetadata.from_outputs(torch.zeros(2, 4), batch_size=2),
        forward_signature=FORWARD_SIGNATURE,
    )

    TuneStrategyTestCorrectness().check_correctness(MockBackend(), "test_model", graph_spec, samples)  # type: ignore

    assert iter_samples.call_args_list == [mocker.call(torch.device("cuda:1")), mocker.call(torch.device("cuda:1"))]


def test_correctness_requires_at_least_one_sample(tmp_path):
    """Correctness validation should not pass without recorded samples."""

    class MockBackend:
        def describe(self):
            return "mock_backend"

    strategy = TuneStrategyTestCorrectness()
    input_spec = output_spec = _input_metadata(((), {}))
    graph_spec = GraphSpec(
        name="test_model",
        input_spec=input_spec,
        output_spec=output_spec,
        forward_signature=FORWARD_SIGNATURE,
    )

    with pytest.raises(ValueError, match="requires at least one sample"):
        strategy.check_correctness(MockBackend(), "test_model", graph_spec, SampleStore.create(tmp_path, "samples"))  # type: ignore


def test_correctness_checks_dynamic_min_and_max_shapes(tmp_path):
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
    input_spec = _input_metadata(recorded_sample, batch_size=2)
    input_spec.update_shapes_seen(_input_metadata(min_sample, batch_size=2))
    input_spec.update_shapes_seen(_input_metadata(max_sample, batch_size=2))
    output_spec = SampleMetadata.from_outputs(torch.randn(2, 8), batch_size=2)
    output_spec.update_shapes_seen(SampleMetadata.from_outputs(torch.randn(2, 4), batch_size=2))
    output_spec.update_shapes_seen(SampleMetadata.from_outputs(torch.randn(2, 12), batch_size=2))
    graph_spec = GraphSpec(
        name="test_model",
        input_spec=input_spec,
        output_spec=output_spec,
        forward_signature=FORWARD_SIGNATURE,
    )

    samples = make_sample_store([recorded_sample], tmp_path)
    strategy.check_correctness(backend, "test_model", graph_spec, samples)  # type: ignore

    assert backend.input_shapes == [(2, 8), (2, 4), (2, 12)]


def test_correctness_skips_value_checks_for_dynamic_boundary_samples(tmp_path):
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
    input_spec = _input_metadata(recorded_sample, batch_size=2)
    input_spec.update_shapes_seen(_input_metadata(min_sample, batch_size=2))
    input_spec.update_shapes_seen(_input_metadata(max_sample, batch_size=2))
    output_spec = SampleMetadata.from_outputs(torch.zeros(2, 8), batch_size=2)
    graph_spec = GraphSpec(
        name="test_model",
        input_spec=input_spec,
        output_spec=output_spec,
        forward_signature=FORWARD_SIGNATURE,
    )

    samples = make_sample_store([recorded_sample], tmp_path)
    strategy.check_correctness(backend, "test_model", graph_spec, samples)  # type: ignore

    assert backend.input_shapes == [(2, 8), (2, 4), (2, 12)]


def test_correctness_reports_dynamic_boundary_inference_failure(tmp_path):
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
    input_spec = _input_metadata(recorded_sample, batch_size=2)
    input_spec.update_shapes_seen(_input_metadata(min_sample, batch_size=2))
    input_spec.update_shapes_seen(_input_metadata(max_sample, batch_size=2))
    output_spec = SampleMetadata.from_outputs(torch.zeros(2, 8), batch_size=2)
    graph_spec = GraphSpec(
        name="test_model",
        input_spec=input_spec,
        output_spec=output_spec,
        forward_signature=FORWARD_SIGNATURE,
    )
    samples = make_sample_store([recorded_sample], tmp_path)

    with pytest.raises(
        CorrectnessDynamicShapeError,
        match="Dynamic shape correctness check failed.*min.*test_model.*mock_backend",
    ):
        strategy.check_correctness(MockBackend(), "test_model", graph_spec, samples)  # type: ignore


def test_correctness_extension_torch_eager_backend_with_nan(torch_device, tmp_path):
    """Test correctness extension with torch eager backend."""

    module = ToyTorchModel()
    graph_spec = module.graph_spec()
    samples = module.samples(device=torch_device)
    samples[0][0][0][0] = float("nan")
    samples = make_sample_store(samples, tmp_path)

    strategy = TuneStrategyTestCorrectness()

    with pytest.raises(CorrectnessValueError, match="contains NaN values"):
        backend = strategy.tune(module, "test_model", graph_spec, samples, torch_device, cache_dir=tmp_path)
        backend.deactivate()


def test_correctness_extension_torch_eager_backend_with_inf(mocker, torch_device, tmp_path):
    """Test correctness extension with torch eager backend."""
    module = ToyTorchModel()
    graph_spec = module.graph_spec()
    samples = module.samples(device=torch_device)
    samples[0][0][0][0] = float("inf")
    samples = make_sample_store(samples, tmp_path)

    mocker.patch.object(module, "forward", return_value=torch.tensor([float("inf")]))

    strategy = TuneStrategyTestCorrectness()

    with pytest.raises(CorrectnessValueError, match="contains infinity values"):
        backend = strategy.tune(module, "test_model", graph_spec, samples, torch_device, cache_dir=tmp_path)
        backend.deactivate()


def test_correctness_extension_torch_eager_backend_with_wrong_shapes(torch_device, tmp_path):
    """Test correctness extension with torch eager backend."""
    module = ToyTorchModel()
    graph_spec = module.graph_spec()
    samples = module.sample_store(tmp_path, device=torch_device, batch_sizes=[1])

    strategy = TuneStrategyTestCorrectness()

    with pytest.raises(
        CorrectnessTensorShapeError, match=r"Expected tensor output to have shape \[2, 5\] but got \[1, 5\]"
    ):
        backend = strategy.tune(module, "test_model", graph_spec, samples, torch_device, cache_dir=tmp_path)
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
    samples = make_sample_store([sample], tmp_path)
    graph_spec = GraphSpec(
        name="test_model",
        input_spec=_input_metadata(sample, batch_size=2),
        output_spec=SampleMetadata.from_outputs(torch.randn(2, 8), batch_size=2),
        forward_signature=FORWARD_SIGNATURE,
    )

    result = strategy._build_and_validate_backend(
        backend,
        module,
        "test_model",
        graph_spec,
        samples,
        torch_device,
        tmp_path,
    )

    assert result is None
    log_text = (tmp_path / "mock_backend_key" / "build.log").read_text(encoding="utf-8")
    assert "Backend build or validation failed" in log_text
    assert "Exception type: RuntimeError" in log_text
    assert "Exception details: correctness failed" in log_text
