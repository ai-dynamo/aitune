# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
"""Unit tests for ONNXExporter."""

from unittest.mock import MagicMock

import pytest

from aitune.torch.backend.tensorrt.onnx_exporter import ONNXExporter
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata
from tests.toy_models import ToyOnnxModel, ToyTorchModel

# Constants for testing
IN_FEATURES = 32
BATCH_SIZE = 2


@pytest.fixture
def mock_torch_onnx(mocker):
    """Fixture that mocks torch.onnx functionality."""
    mock = mocker.patch("aitune.torch.backend.tensorrt.onnx_exporter.torch.onnx")
    mock.export.return_value = mocker.MagicMock()
    return mock


@pytest.fixture
def mock_onnx_lib(mocker):
    """Fixture that mocks onnx library functionality."""
    mock = mocker.patch("aitune.torch.backend.tensorrt.onnx_exporter.onnx")
    mock.load.return_value = mocker.MagicMock()
    mock.checker.check_model.return_value = None
    return mock


def test_onnx_exporter_init(tmp_path):
    """Test ONNXExporter initialization."""
    output_path = tmp_path / "test_model.onnx"

    # Test with default parameters
    exporter = ONNXExporter(output_path=output_path)
    assert exporter.use_dynamo is False
    assert exporter.opset_version is None

    # Test with custom parameters
    output_path2 = tmp_path / "test_model2.onnx"
    exporter = ONNXExporter(output_path=output_path2, use_dynamo=True, opset_version=16)
    assert exporter.use_dynamo is True
    assert exporter.opset_version == 16


def test_export_trace(mock_torch_onnx, mock_onnx_lib, tmp_path):
    """Test export method with standard export."""
    output_path = tmp_path / "onnx" / "test_model.onnx"

    # Create exporter with standard mode (no dynamo)
    exporter = ONNXExporter(output_path=output_path, use_dynamo=False)

    # Create model and sample
    model = ToyTorchModel().eval()
    sample = model.samples(batch_sizes=[BATCH_SIZE], device="cpu")[0]

    args, kwargs = sample
    output = model(*args, **kwargs)

    input_metadata = SampleMetadata.from_inputs(args, kwargs, strict=True)
    output_metadata = SampleMetadata.from_outputs(output, strict=True)

    graph_spec = GraphSpec(
        name="test_graph",
        input_spec=input_metadata,
        output_spec=output_metadata,
    )

    # Export model
    onnx_path = exporter.export(module=model, sample=sample, graph_spec=graph_spec)

    # Verify interactions
    mock_torch_onnx.export.assert_called_once()
    assert "opset_version" in mock_torch_onnx.export.call_args.kwargs
    assert mock_torch_onnx.export.call_args.kwargs["opset_version"] is None

    # Verify verification was called
    mock_onnx_lib.checker.check_model.assert_called_once()

    # Verify returned path
    assert onnx_path == output_path


def test_export_dynamo(mocker, mock_torch_onnx, mock_onnx_lib, tmp_path):
    """Test export method with dynamo export."""
    # Mock dynamo export output
    mock_export_program = MagicMock()
    mock_torch_onnx.export.return_value = mock_export_program

    # Mock graphsurgeon inputs
    mock_graph = mocker.MagicMock()
    mock_graph.inputs = [mocker.MagicMock(name="input__0")]
    mock_graph.outputs = [mocker.MagicMock(name="output__0")]

    output_path = tmp_path / "onnx" / "test_model.onnx"

    # Create exporter with dynamo
    exporter = ONNXExporter(output_path=output_path, use_dynamo=True)

    # Create model and sample
    model = ToyTorchModel().eval()
    sample = model.samples(batch_sizes=[BATCH_SIZE], device="cpu")[0]

    args, kwargs = sample
    output = model(*args, **kwargs)

    input_metadata = SampleMetadata.from_inputs(args, kwargs, strict=True)
    output_metadata = SampleMetadata.from_outputs(output, strict=True)

    graph_spec = GraphSpec(
        name="test_graph",
        input_spec=input_metadata,
        output_spec=output_metadata,
    )

    # Export model
    onnx_path = exporter.export(module=model, sample=sample, graph_spec=graph_spec)

    # Verify interactions
    mock_torch_onnx.export.assert_called_once()
    assert "dynamo" in mock_torch_onnx.export.call_args.kwargs
    assert mock_torch_onnx.export.call_args.kwargs["dynamo"] is True

    # Verify verification was called
    mock_onnx_lib.checker.check_model.assert_called_once()

    # Verify returned path
    assert onnx_path == output_path


def test_export_error_handling(mock_torch_onnx, tmp_path):
    """Test error handling during export."""
    # Set up export to fail
    mock_torch_onnx.export.side_effect = RuntimeError("Export failed")

    output_path = tmp_path / "onnx" / "test_model.onnx"

    # Create exporter
    exporter = ONNXExporter(output_path=output_path)

    # Create model and sample
    model = ToyTorchModel().eval()
    sample = model.samples(batch_sizes=[BATCH_SIZE], device="cpu")[0]

    args, kwargs = sample
    output = model(*args, **kwargs)

    input_metadata = SampleMetadata.from_inputs(args, kwargs, strict=True)
    output_metadata = SampleMetadata.from_outputs(output, strict=True)

    graph_spec = GraphSpec(
        name="test_graph",
        input_spec=input_metadata,
        output_spec=output_metadata,
    )

    # Export should fail
    with pytest.raises(RuntimeError, match="Export failed"):
        exporter.export(
            module=model,
            sample=sample,
            graph_spec=graph_spec,
        )


def test_verify_model(mock_onnx_lib, tmp_path):
    """Test verify_model method."""
    output_path = tmp_path / "test_model.onnx"

    # Create exporter
    exporter = ONNXExporter(output_path=output_path)

    # Verify model
    exporter.verify_model(output_path)

    # Verify interactions
    mock_onnx_lib.checker.check_model.assert_called_once()

    # Test verify failure
    mock_onnx_lib.checker.check_model.side_effect = ValueError("Invalid model")

    with pytest.raises(ValueError, match="Invalid model"):
        exporter.verify_model(output_path)


def test_onnx_exporter_integration(tmp_path):
    """Integration test for ONNXExporter with actual ONNX.

    This test is skipped by default and should be run only in environments
    with ONNX properly installed.
    """
    output_path = tmp_path / "onnx" / "test_model.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create exporter
    exporter = ONNXExporter(output_path=output_path)

    # Create model and sample
    model = ToyTorchModel().eval()
    sample = model.samples(batch_sizes=[BATCH_SIZE], device="cpu")[0]

    args, kwargs = sample
    output = model(*args, **kwargs)

    input_metadata = SampleMetadata.from_inputs(args, kwargs, strict=True)
    output_metadata = SampleMetadata.from_outputs(output, strict=True)

    graph_spec = GraphSpec(
        name="test_graph",
        input_spec=input_metadata,
        output_spec=output_metadata,
    )

    # Export model
    onnx_path = exporter.export(module=model, sample=sample, graph_spec=graph_spec)

    # Verify file exists
    assert onnx_path == output_path

    # Verify model again
    exporter.verify_model(onnx_path)


def test_verify_toy_onnx_model_linear(tmp_path):
    """Test verification of a pre-generated toy linear ONNX model.

    This test uses the ToyOnnxModel function to get the path to a pre-generated
    toy linear ONNX model and verifies that it can be loaded and checked.
    """
    output_path = tmp_path / "dummy_output.onnx"

    # Create exporter
    exporter = ONNXExporter(output_path=output_path)

    # Get path to toy ONNX model (linear by default)
    onnx_path = ToyOnnxModel(is_linear=True).path

    # Verify the model exists
    assert onnx_path.exists(), f"Toy linear ONNX model not found at {onnx_path}"

    # Verify the model
    exporter.verify_model(onnx_path)


def test_verify_toy_onnx_model_conv(tmp_path):
    """Test verification of a pre-generated toy convolutional ONNX model.

    This test uses the ToyOnnxModel function to get the path to a pre-generated
    toy convolutional ONNX model and verifies that it can be loaded and checked.
    """
    output_path = tmp_path / "dummy_output.onnx"

    # Create exporter
    exporter = ONNXExporter(output_path=output_path)

    try:
        # Get path to toy ONNX model (convolutional)
        onnx_path = ToyOnnxModel(is_linear=False).path

        # Verify the model exists
        assert onnx_path.exists(), f"Toy convolutional ONNX model not found at {onnx_path}"

        # Verify the model
        exporter.verify_model(onnx_path)
    except FileNotFoundError:
        pytest.skip("Convolutional toy ONNX model not available")
