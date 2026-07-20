# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ModelOpt calibration."""

import numpy as np
import pytest

from aitune.torch.backend.tensorrt.modelopt_calibration import prepare_calibration_data
from aitune.torch.utils.tensor import format_tensor_name
from tests.toy_models.torch_models import ToyTorchModel
from tests.utilities.helpers import make_graph_spec


def test_prepare_calibration_data():
    torch_model = ToyTorchModel()

    cpu_samples = torch_model.samples()
    graph_spec = torch_model.graph_spec()

    calibration_data = prepare_calibration_data(
        data=cpu_samples,
        graph_spec=graph_spec,
    )

    # ModelOpt format: single dict, ONNX input name -> concatenated array
    input_name = format_tensor_name("x", "input")
    assert input_name in calibration_data
    assert calibration_data[input_name][0].shape == cpu_samples[0][0][0][0].shape
    assert calibration_data[input_name][1].shape == cpu_samples[0][0][0][1].shape
    assert len(calibration_data[input_name]) == len(cpu_samples[0][0][0]) == 2  # two batched samples


def test_prepare_calibration_data_returns_numpy_dict():
    """Result is a dict of numpy arrays with expected dtypes."""
    model = ToyTorchModel()
    samples = model.samples()
    graph_spec = model.graph_spec()

    calibration_data = prepare_calibration_data(data=samples, graph_spec=graph_spec)

    assert isinstance(calibration_data, dict)
    for name, arr in calibration_data.items():
        assert isinstance(arr, np.ndarray), f"{name!r} should be numpy array"
        assert arr.dtype == np.float32, f"{name!r} should be float32"


def test_prepare_calibration_data_multiple_samples_concatenated():
    """Multiple samples are concatenated along axis=0."""
    model = ToyTorchModel()
    # Three samples with batch sizes 1, 2, 3 -> total 6 rows
    samples = model.samples(batch_sizes=[1, 2, 3])
    graph_spec = model.graph_spec()

    calibration_data = prepare_calibration_data(data=samples, graph_spec=graph_spec)

    assert calibration_data[format_tensor_name("x", "input")].shape == (6, 32)


def test_prepare_calibration_data_empty_data_raises():
    """Empty data list raises ValueError."""
    model = ToyTorchModel()
    graph_spec = model.graph_spec()

    with pytest.raises(ValueError, match="must not be empty"):
        prepare_calibration_data(data=[], graph_spec=graph_spec)


def test_prepare_calibration_data_graph_spec_no_tensors_raises():
    """GraphSpec with no tensor inputs raises ValueError."""
    graph_spec = make_graph_spec(lambda: None, ((), {}), name="empty")

    with pytest.raises(ValueError, match="no tensor inputs"):
        prepare_calibration_data(data=[((), {})], graph_spec=graph_spec)


def test_prepare_calibration_data_all_samples_skip_no_tensors_raises():
    """Samples with no tensors at expected locators raise ValueError."""
    model = ToyTorchModel()
    graph_spec = model.graph_spec()
    # Pass scalars instead of tensors so no tensor is found for x
    samples = [((1,), {}), ((2,), {})]

    with pytest.raises(ValueError, match="No valid calibration samples"):
        prepare_calibration_data(data=samples, graph_spec=graph_spec)
