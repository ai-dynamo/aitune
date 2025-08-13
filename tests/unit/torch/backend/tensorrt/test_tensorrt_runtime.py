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
"""Unit tests for TensorRTRuntime."""

import pytest

from aitune.torch.backend.tensorrt.tensorrt_runtime import (
    TensorRTRuntime,
)


@pytest.fixture
def mock_trt(mocker):
    """Fixture that mocks TensorRT functionality."""
    mock_trt = mocker.patch("aitune.torch.backend.tensorrt.tensorrt_runtime.trt")

    # Set up TensorIOMode enum
    class MockTensorIOMode:
        INPUT = 0
        OUTPUT = 1

    mock_trt.TensorIOMode = MockTensorIOMode

    # Create mock runtime
    mock_runtime = mocker.MagicMock()
    mock_trt.Runtime.return_value = mock_runtime

    # Create mock engine from runtime
    mock_engine = mocker.MagicMock()
    mock_runtime.deserialize_cuda_engine.return_value = mock_engine

    # Create mock execution context from engine
    mock_context = mocker.MagicMock()
    mock_engine.create_execution_context.return_value = mock_context

    # Set up other mock components
    mock_trt.Logger.WARNING = 1
    mock_trt.Logger.return_value = mocker.MagicMock()

    # Set up tensor methods to match create_execution_context implementation
    tensor_names = ["input", "output"]
    # We need to set num_io_tensors and get_tensor_name
    mock_engine.num_io_tensors = len(tensor_names)
    mock_engine.get_tensor_name.side_effect = lambda i: tensor_names[i]

    # Configure tensor modes, shapes and dtypes
    mock_engine.get_tensor_mode.side_effect = lambda name: (
        MockTensorIOMode.INPUT if name == "input" else MockTensorIOMode.OUTPUT
    )
    mock_engine.get_tensor_dtype.return_value = "float32"  # Any dtype will do for testing
    mock_context.get_tensor_shape.return_value = [1, 3, 224, 224]  # Example shape

    return mock_trt, mock_engine, mock_context


def test_create_execution_context(mock_trt, mocker):
    """Test create_execution_context method."""

    mock_trt_module, mock_engine, mock_context = mock_trt
    # Mock TensorRTEngineInfo
    mock_engine_info = mocker.MagicMock()
    mock_engine_info_class = mocker.patch(
        "aitune.torch.backend.tensorrt.tensorrt_runtime.TensorRTEngineInfo", return_value=mock_engine_info
    )

    # Create builder
    runtime = TensorRTRuntime()

    # Create mock engine bytes
    engine_bytes = b"mock engine bytes"

    # Create execution context
    context, bindings, input_names, output_names, engine_info = runtime.create_execution_context(engine_bytes)

    # Verify TensorRT interactions
    mock_trt_module.Runtime.assert_called_once()
    mock_trt_module.Runtime().deserialize_cuda_engine.assert_called_once_with(engine_bytes)
    mock_engine.create_execution_context.assert_called_once()

    # Verify returned values
    assert context is mock_context
    assert isinstance(bindings, dict)
    # Verify input and output names are populated
    assert len(input_names) > 0
    assert len(output_names) > 0
    # We expect 'input' to be in the input names and 'output' to be in output names
    # due to our fixture setup

    assert "input" in input_names
    assert "output" in output_names
    # Verify TensorRTEngineInfo was created and returned with the engine
    mock_engine_info_class.assert_called_once_with(engine=mock_engine)
    assert engine_info is mock_engine_info


def test_load_engine(tmp_path):
    """Test load_engine method."""
    # Create mock engine file
    engine_content = b"mock engine content"
    engine_path = tmp_path / "tensorrt" / "test_model.plan"
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    with open(engine_path, "wb") as f:
        f.write(engine_content)

    # Create builder
    runtime = TensorRTRuntime()

    # Load engine
    loaded_engine = runtime.load_engine(engine_path)

    # Verify loaded content
    assert loaded_engine == engine_content


def test_load_engine_file_not_found(tmp_path):
    """Test load_engine method with non-existent engine file."""
    # Create builder
    runtime = TensorRTRuntime()

    # Engine path that doesn't exist
    engine_path = tmp_path / "nonexistent.plan"

    # Load should fail
    with pytest.raises(FileNotFoundError):
        runtime.load_engine(engine_path)
