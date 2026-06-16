# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for correctness checking functionality."""

import numpy as np
import pytest
import torch

from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.module.tensor_spec import TensorSpec
from aitune.torch.task.correctness import (
    CorrectnessDynamicShapeError,
    CorrectnessTensorShapeError,
    _check_output_correctness,
    _check_output_tensor_shapes,
    _dynamic_shape_boundary_samples,
    check_dynamic_shape_boundary_inference,
    check_inference_output_correctness,
)


def test_private_check_output_correctness_valid():
    """Test _check_output_correctness with valid outputs."""
    outputs = {"output1": torch.tensor([1.0, 2.0, 3.0]), "output2": np.array([4.0, 5.0, 6.0])}
    _check_output_correctness(outputs)


def test_private_check_output_correctness_nan():
    """Test _check_output_correctness with NaN values."""
    outputs = {"output1": torch.tensor([1.0, float("nan"), 3.0]), "output2": np.array([4.0, 5.0, 6.0])}
    with pytest.raises(ValueError, match="contains NaN values"):
        _check_output_correctness(outputs)


def test_private_check_output_correctness_inf():
    """Test _check_output_correctness with infinity values."""
    outputs = {"output1": torch.tensor([1.0, 2.0, 3.0]), "output2": np.array([4.0, float("inf"), 6.0])}
    with pytest.raises(ValueError, match="contains infinity values"):
        _check_output_correctness(outputs)


def test_private_check_output_correctness_just_a_string():
    """Test _check_output_correctness with just a string."""
    outputs = {
        "output1": "not a tensor",
    }
    _check_output_correctness(outputs)


def test_private_check_output_correctness_list():
    """Test _check_output_correctness with a list of tensors and numpy arrays."""
    outputs = [torch.tensor([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    _check_output_correctness(outputs)


def test_private_check_output_correctness_sample():
    """Test _check_output_correctness with a sample output."""
    outputs = (
        (torch.tensor([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])),
        {"output1": torch.tensor([1.0, 2.0, 3.0]), "output2": np.array([4.0, 5.0, 6.0])},
    )
    _check_output_correctness(outputs)


def test_private_check_output_correctness_sample_scalar_nan():
    """Test _check_output_correctness with a scalar NaN sample output."""
    outputs = (
        (float("nan"), torch.tensor([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])),
        {"output1": torch.tensor([1.0, 2.0, 3.0]), "output2": np.array([4.0, 5.0, 6.0])},
    )
    with pytest.raises(ValueError, match="is not finite"):
        _check_output_correctness(outputs)


def test_check_inference_output_correctness_deepcopies_inputs():
    """Recorded-sample correctness should not mutate stored samples."""
    cache = []
    data = [((cache,), {"cache": cache})]
    output_spec = SampleMetadata.from_outputs(torch.zeros(2, 4), batch_size=2)

    def infer(*args, **kwargs):
        args[0].append("not important, should be discarded")
        kwargs["cache"].append("not important, should be discarded")
        return torch.zeros(2, 4)

    check_inference_output_correctness(data, output_spec, infer=infer, name="test_model.mock_backend")

    assert cache == []


def test_private_dynamic_shape_boundary_samples_resize_args_and_kwargs():
    """Boundary samples should cover configured min/max input shapes from one stored sample."""
    args = (torch.arange(16, dtype=torch.float32).reshape(2, 8),)
    kwargs = {"mask": torch.arange(16, dtype=torch.float32).reshape(2, 8)}
    input_spec = SampleMetadata.from_inputs(args, kwargs, batch_size=2)

    min_args = (torch.randn(2, 4),)
    min_kwargs = {"mask": torch.randn(2, 4)}
    max_args = (torch.randn(2, 12),)
    max_kwargs = {"mask": torch.randn(2, 12)}
    input_spec.update_shapes_seen(SampleMetadata.from_inputs(min_args, min_kwargs, batch_size=2))
    input_spec.update_shapes_seen(SampleMetadata.from_inputs(max_args, max_kwargs, batch_size=2))

    min_sample, max_sample = _dynamic_shape_boundary_samples((args, kwargs), input_spec)

    assert min_sample[0][0].shape == torch.Size([2, 4])
    assert min_sample[1]["mask"].shape == torch.Size([2, 4])
    assert max_sample[0][0].shape == torch.Size([2, 12])
    assert max_sample[1]["mask"].shape == torch.Size([2, 12])


def test_private_dynamic_shape_boundary_samples_returns_empty_for_static_shapes():
    """Static input specs should not add boundary correctness samples."""
    args = (torch.randn(2, 6),)
    input_spec = SampleMetadata.from_inputs(args, {}, batch_size=2)

    assert _dynamic_shape_boundary_samples((args, {}), input_spec) == []


def test_check_dynamic_shape_boundary_inference_reports_backend_failure():
    """Dynamic boundary inference failures should be raised as correctness errors."""
    args = (torch.randn(2, 8),)
    input_spec = SampleMetadata.from_inputs(args, {}, batch_size=2)
    input_spec.update_shapes_seen(SampleMetadata.from_inputs((torch.randn(2, 4),), {}, batch_size=2))
    input_spec.update_shapes_seen(SampleMetadata.from_inputs((torch.randn(2, 12),), {}, batch_size=2))
    output_spec = SampleMetadata.from_outputs(torch.zeros(2, 8), batch_size=2)

    def infer(tensor):
        if tuple(tensor.shape) != (2, 8):
            raise RuntimeError("backend shape failure")
        return tensor

    with pytest.raises(
        CorrectnessDynamicShapeError,
        match="Dynamic shape correctness check failed.*min.*test_model.mock_backend",
    ) as exc_info:
        check_dynamic_shape_boundary_inference(
            (args, {}), input_spec, output_spec, infer=infer, name="test_model.mock_backend"
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_check_dynamic_shape_boundary_inference_validates_output_shapes_without_value_check():
    """Dynamic boundary outputs should be shape-checked without finite-value checks."""
    args = (torch.randn(2, 8),)
    input_spec = SampleMetadata.from_inputs(args, {}, batch_size=2)
    input_spec.update_shapes_seen(SampleMetadata.from_inputs((torch.randn(2, 4),), {}, batch_size=2))
    input_spec.update_shapes_seen(SampleMetadata.from_inputs((torch.randn(2, 12),), {}, batch_size=2))
    output_spec = SampleMetadata.from_outputs(torch.zeros(2, 8), batch_size=2)
    output_spec.update_shapes_seen(SampleMetadata.from_outputs(torch.zeros(2, 4), batch_size=2))
    output_spec.update_shapes_seen(SampleMetadata.from_outputs(torch.zeros(2, 12), batch_size=2))

    def infer(tensor):
        return torch.full(tuple(tensor.shape), float("nan"))

    check_dynamic_shape_boundary_inference(
        (args, {}), input_spec, output_spec, infer=infer, name="test_model.mock_backend"
    )


def test_check_dynamic_shape_boundary_inference_reports_output_shape_failure():
    """Dynamic boundary inference should fail when outputs do not match the graph output spec."""
    args = (torch.randn(2, 8),)
    input_spec = SampleMetadata.from_inputs(args, {}, batch_size=2)
    input_spec.update_shapes_seen(SampleMetadata.from_inputs((torch.randn(2, 4),), {}, batch_size=2))
    input_spec.update_shapes_seen(SampleMetadata.from_inputs((torch.randn(2, 12),), {}, batch_size=2))
    output_spec = SampleMetadata.from_outputs(torch.zeros(2, 8), batch_size=2)

    def infer(tensor):
        return torch.zeros(tensor.shape[0], tensor.shape[1], 1)

    with pytest.raises(CorrectnessTensorShapeError, match="output tensor shapes"):
        check_dynamic_shape_boundary_inference(
            (args, {}), input_spec, output_spec, infer=infer, name="test_model.mock_backend"
        )


def test_check_dynamic_shape_boundary_inference_reports_missing_output_data():
    """Dynamic boundary inference should fail when the backend returns no output tensors."""
    args = (torch.randn(2, 8),)
    input_spec = SampleMetadata.from_inputs(args, {}, batch_size=2)
    input_spec.update_shapes_seen(SampleMetadata.from_inputs((torch.randn(2, 4),), {}, batch_size=2))
    input_spec.update_shapes_seen(SampleMetadata.from_inputs((torch.randn(2, 12),), {}, batch_size=2))
    output_spec = SampleMetadata.from_outputs(torch.zeros(2, 8), batch_size=2)

    def infer(tensor):
        del tensor
        return None

    with pytest.raises(CorrectnessTensorShapeError, match="Expected 1 output tensor"):
        check_dynamic_shape_boundary_inference(
            (args, {}), input_spec, output_spec, infer=infer, name="test_model.mock_backend"
        )


def test_private_check_output_tensor_shapes_matching_shapes():
    """Test _check_output_tensor_shapes with matching tensor shapes."""
    expected_specs = [
        TensorSpec.from_tensor("output__0", torch.randn(2, 5), batch_size=2),
        TensorSpec.from_tensor("output__1", torch.randn(2, 10), batch_size=2),
    ]
    actual_specs = [
        TensorSpec.from_tensor("output__0", torch.randn(2, 5), batch_size=2),
        TensorSpec.from_tensor("output__1", torch.randn(2, 10), batch_size=2),
    ]

    # Should not raise any exception
    _check_output_tensor_shapes(expected_specs, actual_specs)


def test_private_check_output_tensor_shapes_matching_with_symbolic_dimensions():
    """Test _check_output_tensor_shapes with symbolic dimensions that should match."""
    # Create specs with symbolic dimensions that should match
    expected_specs = [
        TensorSpec(
            name="output__0",
            shape=[2, "dim1"],
            min_shape=[2, 5],
            max_shape=[2, 10],
            dtype=torch.float32,
            _bs_multipliers=[1.0, 2.5],
        ),
        TensorSpec(
            name="output__1",
            shape=[2, "batch1"],
            min_shape=[2, 10],
            max_shape=[2, 10],
            dtype=torch.float32,
            _bs_multipliers=[1.0, 5.0],
        ),
    ]
    actual_specs = [
        TensorSpec(
            name="output__0",
            shape=[2, 7],  # Concrete dimension that falls within symbolic range
            min_shape=[2, 7],
            max_shape=[2, 7],
            dtype=torch.float32,
            _bs_multipliers=[1.0, 3.5],
        ),
        TensorSpec(
            name="output__1",
            shape=[2, 10],  # Concrete dimension that matches batch dimension
            min_shape=[2, 10],
            max_shape=[2, 10],
            dtype=torch.float32,
            _bs_multipliers=[1.0, 5.0],
        ),
    ]

    # Should not raise any exception
    _check_output_tensor_shapes(expected_specs, actual_specs)


def test_private_check_output_tensor_shapes_mismatched_shapes():
    """Test _check_output_tensor_shapes with mismatched tensor shapes."""
    expected_specs = [
        TensorSpec.from_tensor("output__0", torch.randn(2, 5), batch_size=2),
        TensorSpec.from_tensor("output__1", torch.randn(2, 10), batch_size=2),
        TensorSpec.from_tensor("output__2", torch.randn(2, 10, 20), batch_size=2),
    ]
    actual_specs = [
        TensorSpec.from_tensor("output__0", torch.randn(1, 5), batch_size=1),  # Different batch size
        TensorSpec.from_tensor("output__1", torch.randn(2, 8), batch_size=2),  # Different feature size
        TensorSpec.from_tensor("output__2", torch.randn(2, 10), batch_size=2),
    ]

    with pytest.raises(CorrectnessTensorShapeError) as exc_info:
        _check_output_tensor_shapes(expected_specs, actual_specs)

    error_message = str(exc_info.value)
    assert "Expected tensor output__0 to have shape [2, 5] but got [1, 5]" in error_message
    assert "Expected tensor output__1 to have shape [2, 10] but got [2, 8]" in error_message
    assert "Expected tensor output__2 to have shape [2, 10, 20] but got [2, 10]" in error_message
    assert "3 error(s) related to output tensor shapes" in error_message
