# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit test for recording module."""

from unittest.mock import Mock

import pytest
import torch

from aitune.exceptions import AITuneUserInputError
from aitune.torch.config import AITuneConfig
from aitune.torch.dynamic_shapes import BatchDim, DynamicDim
from aitune.torch.module.recording_module import RecordingModule
from aitune.torch.module.sample_store import SampleStore
from tests.toy_models.torch_models import ToyTorchModel


def recording_module(strict_mode, tmp_path):
    config = AITuneConfig()
    config.max_num_samples_stored = 10
    config.min_num_samples = 2
    config.strict_mode = strict_mode

    module = Mock(spec=torch.nn.Module)
    module.__call__ = lambda *args, **kwargs: args[0]
    return RecordingModule(module, "test-module", config, cache_dir_resolver=lambda: tmp_path)


@pytest.mark.parametrize("strict_mode", [True, False])
def test_recording_module_same_rank_tensors(strict_mode, tmp_path):
    rec_module = recording_module(strict_mode=strict_mode, tmp_path=tmp_path)
    rec_module(torch.tensor(2))
    rec_module(torch.tensor(2))

    assert len(rec_module.graph_specs) == 1


def test_recording_module_same_other_data(tmp_path):
    rec_module = recording_module(strict_mode=True, tmp_path=tmp_path)
    rec_module(1)
    rec_module(1)

    assert len(rec_module.graph_specs) == 1


@pytest.mark.parametrize("strict_mode", [True, False])
def test_recording_module_call_different_rank_tensors(strict_mode, tmp_path):
    rec_module = recording_module(strict_mode=strict_mode, tmp_path=tmp_path)
    rec_module(torch.tensor(2))
    rec_module(torch.tensor([1, 1]))

    assert len(rec_module.graph_specs) == 2


def test_recording_module_same_rank_tensors_different_other_data(tmp_path):
    rec_module = recording_module(strict_mode=True, tmp_path=tmp_path)
    rec_module(torch.tensor(2), "abc")
    rec_module(torch.tensor(2), "xyz")

    assert len(rec_module.graph_specs) == 2


def test_recording_module_call_multiple_graphs(tmp_path):
    rec_module = recording_module(strict_mode=False, tmp_path=tmp_path)
    rec_module(torch.tensor(42))  # first graph - dtype int
    rec_module(torch.randn(2))  # second graph - dtype fp32
    rec_module(torch.randn(3))  # second graph batch dimension
    rec_module(torch.randn(1, 1), torch.randn(2, 1))  # third graph
    rec_module(torch.randn(1, 3), torch.randn(2, 3))  # third graph batched dimensions

    assert rec_module.record_sample
    assert len(rec_module.graph_specs) == 3
    assert rec_module.graph_specs[0].input_spec.tensor_specs[0].shape == []
    assert rec_module.graph_specs[1].input_spec.tensor_specs[0].shape == ["dim0"]
    assert rec_module.graph_specs[2].input_spec.tensor_specs[0].shape == [1, "dim1"]
    assert rec_module.graph_specs[2].input_spec.tensor_specs[0].min_shape == [1, 1]
    assert rec_module.graph_specs[2].input_spec.tensor_specs[0].max_shape == [1, 3]
    assert rec_module.graph_specs[2].input_spec.tensor_specs[1].min_shape == [2, 1]
    assert rec_module.graph_specs[2].input_spec.tensor_specs[1].max_shape == [2, 3]


def test_recording_module_check_is_ready_non_strict(tmp_path):
    rec_module = recording_module(strict_mode=False, tmp_path=tmp_path)
    rec_module(torch.tensor(1))  # first graph, one sample
    assert not rec_module.is_ready_for_optimization

    rec_module(torch.randn(1, 1))  # second graph, one sample
    assert not rec_module.is_ready_for_optimization

    rec_module(torch.tensor(2))  # first graph, second sample
    rec_module(torch.randn(1, 1))  # second graph, second sample
    assert rec_module.is_ready_for_optimization


def test_samples_for_graph(tmp_path):
    rec_module = recording_module(strict_mode=True, tmp_path=tmp_path)
    rec_module(1, torch.tensor(1))  # first graph
    rec_module(1, torch.tensor(1), a=5)  # second graph
    rec_module(2, torch.tensor(1))  # third graph
    rec_module(2, torch.tensor(1))  # third graph

    for graph_spec, num_expected_samples in zip(rec_module.graph_specs, [1, 1, 2], strict=False):
        samples = rec_module.samples_for_graph_spec(graph_spec)
        assert isinstance(samples, SampleStore)
        assert len(samples) == num_expected_samples

    samples = rec_module.samples_for_graph_spec(rec_module.graph_specs[1])
    args, kwargs = samples[0]  # take the only sample
    assert args == (1, torch.tensor(1))
    assert kwargs == {"a": 5}


def test_samples_are_isolated_from_graph_cache_artifacts(tmp_path):
    cache_dir = tmp_path / "module"
    graph_cache_dir = cache_dir / "0"
    graph_cache_dir.mkdir(parents=True)
    backend_artifact = graph_cache_dir / "build.log"
    backend_artifact.write_text("existing backend artifact")
    recording = RecordingModule(torch.nn.Identity(), "identity", cache_dir_resolver=lambda: cache_dir)

    recording(torch.tensor([1.0]))

    graph_spec = recording.graph_specs[0]
    store = recording.samples_for_graph_spec(graph_spec)
    assert store.to_dict()["artifact"].path == graph_cache_dir / "samples"
    assert backend_artifact.read_text() == "existing backend artifact"


def test_cache_dir_resolver_is_called_when_graph_is_recorded(tmp_path):
    cache_dir_resolver = Mock(return_value=tmp_path)
    recording = RecordingModule(torch.nn.Identity(), "identity", cache_dir_resolver=cache_dir_resolver)

    cache_dir_resolver.assert_not_called()
    recording(torch.tensor([1.0]))

    cache_dir_resolver.assert_called_once_with()


def test_device(tmp_path):
    model = ToyTorchModel()
    recording = RecordingModule(model, "test-module", cache_dir_resolver=lambda: tmp_path)
    assert recording.device == torch.device("cpu")


def test_recording_stores_original_args_and_kwargs(tmp_path):
    """Test that recording stores original args and kwargs before calling the model to avoid side effects."""

    class Model(torch.nn.Module):
        def forward(self, *args, **kwargs):
            args[0].append("not important, should be discarded")
            kwargs["cache"] = "not important, should be discarded"
            return args, kwargs

    model = Model()
    recording = RecordingModule(model, "test-module", cache_dir_resolver=lambda: tmp_path)
    recording([], cache=[])

    assert len(recording.graph_specs) == 1
    samples = recording.samples_for_graph_spec(recording.graph_specs[0])
    assert len(samples) == 1
    args, kwargs = samples[0]
    assert args[0] == []
    assert kwargs["cache"] == []


def test_recording_tracks_input_metadata_before_and_after_call(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, x, cache, feat_idx):
            cache[0] = torch.ones(2, 3)
            feat_idx[0] += 1
            return x

    config = AITuneConfig()
    config.strict_mode = True
    recording = RecordingModule(Model(), "test-module", config, cache_dir_resolver=lambda: tmp_path)

    recording(torch.ones(1), [None], [0])

    graph_spec = recording.graph_specs[0]
    pre_other = {locator.path: value for locator, value in graph_spec.input_spec.other_data}
    post_other = {locator.path: value for locator, value in graph_spec.post_input_spec.other_data}
    post_tensors = {locator.path: spec for locator, spec in graph_spec.post_input_spec.tensor_data}

    assert pre_other[("cache", 0)] is None
    assert pre_other[("feat_idx", 0)] == 0
    assert post_other[("feat_idx", 0)] == 1
    assert post_tensors[("cache", 0)].shape == [2, 3]

    stored_args, _ = recording.samples_for_graph_spec(graph_spec)[0]
    assert stored_args[1] == [None]
    assert stored_args[2] == [0]


def test_recording_normalizes_equivalent_call_layouts(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    config = AITuneConfig()
    config.max_num_samples_stored = 4
    recording = RecordingModule(Model(), "test-module", config, cache_dir_resolver=lambda: tmp_path)
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)

    recording(x, y)
    recording(x, y=y)
    recording(x=x, y=y)
    recording(y=y, x=x)

    assert len(recording.graph_specs) == 1
    samples = recording.samples_for_graph_spec(recording.graph_specs[0])
    assert len(samples) == 4
    assert all(len(args) == 2 and kwargs == {} for args, kwargs in samples)


def test_recording_limits_stored_samples_without_limiting_metadata(tmp_path):
    config = AITuneConfig()
    config.max_num_samples_stored = 2
    config.strict_mode = False
    recording = RecordingModule(torch.nn.Identity(), "identity", config, cache_dir_resolver=lambda: tmp_path)
    copy_inputs = Mock(wraps=recording._copy_inputs)
    recording._copy_inputs = copy_inputs

    for size in range(1, 5):
        recording(torch.arange(size))

    graph_spec = recording.graph_specs[0]
    samples = recording.samples_for_graph_spec(graph_spec)
    sample_dir = samples.to_dict()["artifact"].path
    assert len(samples) == 2
    assert {path.name for path in sample_dir.iterdir()} == {"sample-00000.pt", "sample-00001.pt"}
    assert [sample[0][0].shape for sample in samples] == [torch.Size([1]), torch.Size([2])]
    assert copy_inputs.call_count == 4
    assert graph_spec.input_spec.tensor_specs[0].max_shape == [4]


def test_recording_raises_error_if_model_is_not_in_no_grad_mode(tmp_path):
    """Test that recording raises an error if the model is not in no_grad mode."""
    rec_module = recording_module(strict_mode=True, tmp_path=tmp_path)

    normal_input = torch.tensor(1.0)
    rec_module(normal_input)  # no exception

    requires_grad_input = torch.tensor(1.0, requires_grad=True) + 1  # +1 fills grad buffer
    with pytest.raises(RuntimeError, match=r"Cannot copy model inputs\. Model is not in no_grad mode\."):
        rec_module(requires_grad_input)


def test_recording_resolves_nested_shapes_and_shared_dimensions(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, input_ids, options, extra):
            return input_ids + options["mask"] + extra[:, : input_ids.shape[1]]

    dynamic_shapes = {
        "input_ids": (BatchDim("batch", min=1, opt=2, max=4), DynamicDim("sequence", min=2, opt=4, max=8)),
        ("options", "mask"): (
            BatchDim("batch", min=1, opt=2, max=4),
            DynamicDim("sequence", min=2, opt=4, max=8),
        ),
    }
    recording = RecordingModule(
        Model(),
        "encoder",
        dynamic_shapes=dynamic_shapes,
        cache_dir_resolver=lambda: tmp_path,
    )

    recording(torch.ones(2, 4), {"mask": torch.ones(2, 4)}, torch.ones(2, 5))
    recording(torch.ones(2, 4), {"mask": torch.ones(2, 4)}, torch.ones(2, 7))

    graph_spec = recording.graph_specs[0]
    assert graph_spec.dynamic_shapes == dynamic_shapes
    tensor_specs = {locator.path: spec for locator, spec in graph_spec.input_spec.tensor_data}
    assert tensor_specs["extra"].max_shape == [2, 7]

    with pytest.raises(AITuneUserInputError, match="must have the same size"):
        recording(torch.ones(2, 4), {"mask": torch.ones(2, 5)}, torch.ones(2, 5))


@pytest.mark.parametrize(
    ("definition", "shape", "message"),
    [
        ((DynamicDim("size", min=1, max=4),), (2, 3), "expects rank 1"),
        ((2, 3), (2, 4), "expects size 3"),
        ((DynamicDim("size", min=2, max=4),), (1,), "between 2 and 4"),
    ],
)
def test_recording_rejects_shapes_outside_the_definition(definition, shape, message, tmp_path):
    recording = RecordingModule(
        torch.nn.Identity(),
        "identity",
        dynamic_shapes={"input": definition},
        cache_dir_resolver=lambda: tmp_path,
    )

    with pytest.raises(AITuneUserInputError, match=message):
        recording(torch.ones(shape))


def test_recording_rejects_configured_paths_that_were_never_observed(tmp_path):
    recording = RecordingModule(
        torch.nn.Identity(),
        "identity",
        dynamic_shapes={"missing": (DynamicDim("size", min=1, max=4),)},
        cache_dir_resolver=lambda: tmp_path,
    )
    recording(torch.ones(2))

    with pytest.raises(AITuneUserInputError, match="did not match any recorded tensor"):
        recording.validate_dynamic_shape_paths_recorded()
