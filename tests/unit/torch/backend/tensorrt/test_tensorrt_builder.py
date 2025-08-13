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
"""Unit tests for TensorRTBuilder."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from aitune.torch.backend.tensorrt.tensorrt_builder import (
    TensorRTBuilder,
)
from aitune.torch.backend.tensorrt.tensorrt_runtime import TensorRTRuntime
from aitune.torch.utils.cuda import is_available as is_cuda_available
from tests.toy_models import ToyOnnxModel


@pytest.fixture
def mock_polygraphy(mocker):
    """Fixture that mocks Polygraphy functionality."""
    mock_network = mocker.patch("aitune.torch.backend.tensorrt.tensorrt_builder.network_from_onnx_path")
    mock_create_config = mocker.patch("aitune.torch.backend.tensorrt.tensorrt_builder.CreateConfig")
    mock_engine = mocker.patch("aitune.torch.backend.tensorrt.tensorrt_builder.engine_from_network")
    mock_save = mocker.patch("aitune.torch.backend.tensorrt.tensorrt_builder.save_engine")
    mock_profile = mocker.patch("aitune.torch.backend.tensorrt.tensorrt_builder.Profile")

    mock_network.return_value = mocker.MagicMock()
    mock_create_config.return_value = mocker.MagicMock()
    mock_engine.return_value = mocker.MagicMock()
    mock_save.return_value = None
    mock_profile_instance = mocker.MagicMock()
    mock_profile.return_value = mock_profile_instance

    return {
        "network": mock_network,
        "config": mock_create_config,
        "engine": mock_engine,
        "save": mock_save,
        "profile": mock_profile,
        "profile_instance": mock_profile_instance,
    }


@pytest.fixture
def mock_trt(mocker):
    """Fixture that mocks TensorRT functionality."""
    mock_trt = mocker.patch("aitune.torch.backend.tensorrt.tensorrt_builder.trt")

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


def test_tensorrt_builder_init(tmp_path):
    """Test TensorRTBuilder initialization."""
    input_onnx_path = tmp_path / "input_model.onnx"
    output_path = tmp_path / "output_model.plan"

    # Test with default parameters
    builder = TensorRTBuilder(input_onnx_path=input_onnx_path, output_path=output_path)
    assert builder.workspace_size is None
    assert builder.min_shapes is None
    assert builder.opt_shapes is None
    assert builder.max_shapes is None
    assert builder.optimization_level is None
    assert builder.compatibility_level is None
    assert builder.timing_cache is None

    # Test with custom parameters
    input_onnx_path2 = tmp_path / "input_model2.onnx"
    output_path2 = tmp_path / "output_model2.plan"
    builder = TensorRTBuilder(
        input_onnx_path=input_onnx_path2,
        output_path=output_path2,
        workspace_size=2 << 30,  # 2GB
        optimization_level=5,
        compatibility_level=0,
        timing_cache=Path("/tmp/timing.cache"),
        min_shapes={"input": (1, 3, 224, 224)},
        opt_shapes={"input": (4, 3, 224, 224)},
        max_shapes={"input": (8, 3, 224, 224)},
    )
    assert builder.workspace_size == 2 << 30
    assert builder.optimization_level == 5
    assert builder.timing_cache == Path("/tmp/timing.cache")
    assert builder.min_shapes == {"input": (1, 3, 224, 224)}
    assert builder.opt_shapes == {"input": (4, 3, 224, 224)}
    assert builder.max_shapes == {"input": (8, 3, 224, 224)}


def test_build(mock_polygraphy, tmp_path):
    """Test build method."""
    # Create mock ONNX path
    onnx_path = tmp_path / "test_model.onnx"
    with open(onnx_path, "w") as f:
        f.write("mock onnx content")

    # Create builder
    engine_path = tmp_path / "tensorrt" / "test_model.plan"
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    builder = TensorRTBuilder(input_onnx_path=onnx_path, output_path=engine_path)

    # Build engine
    returned_path = builder.build()

    # Verify interactions
    mock_polygraphy["network"].assert_called_once_with(str(onnx_path), strongly_typed=True)
    mock_polygraphy["config"].assert_called_once()
    mock_polygraphy["engine"].assert_called_once_with(
        network=mock_polygraphy["network"].return_value, config=mock_polygraphy["config"].return_value
    )
    mock_polygraphy["save"].assert_called_once_with(
        engine=mock_polygraphy["engine"].return_value, path=str(engine_path)
    )

    # Verify returned path
    assert returned_path == engine_path


def test_build_with_dynamic_shapes(mock_polygraphy, tmp_path):
    """Test build method with dynamic shapes."""
    # Create mock ONNX path
    onnx_path = tmp_path / "test_model.onnx"
    with open(onnx_path, "w") as f:
        f.write("mock onnx content")

    # Create shapes for testing
    min_shapes = {"input": (1, 3, 224, 224)}
    opt_shapes = {"input": (4, 3, 224, 224)}
    max_shapes = {"input": (8, 3, 224, 224)}

    # Create builder with dynamic shapes
    engine_path = tmp_path / "tensorrt" / "test_model.plan"
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    builder = TensorRTBuilder(
        input_onnx_path=onnx_path,
        output_path=engine_path,
        min_shapes=min_shapes,
        opt_shapes=opt_shapes,
        max_shapes=max_shapes,
    )

    # Build engine
    returned_path = builder.build()

    # Verify profile interactions
    mock_polygraphy["profile"].assert_called_once()
    mock_polygraphy["profile_instance"].add.assert_called_once_with(
        name="input", min=min_shapes["input"], opt=opt_shapes["input"], max=max_shapes["input"]
    )

    # Verify build interactions
    mock_polygraphy["network"].assert_called_once_with(str(onnx_path), strongly_typed=True)
    mock_polygraphy["config"].assert_called_once()
    mock_polygraphy["engine"].assert_called_once_with(
        network=mock_polygraphy["network"].return_value, config=mock_polygraphy["config"].return_value
    )
    mock_polygraphy["save"].assert_called_once_with(
        engine=mock_polygraphy["engine"].return_value, path=str(engine_path)
    )

    # Verify returned path
    assert returned_path == engine_path


def test_build_file_not_found(tmp_path):
    """Test build method with non-existent ONNX file."""
    # ONNX path that doesn't exist
    onnx_path = tmp_path / "nonexistent.onnx"
    engine_path = tmp_path / "tensorrt" / "test_model.plan"

    # Create builder
    builder = TensorRTBuilder(input_onnx_path=onnx_path, output_path=engine_path)

    # Build should fail
    with pytest.raises(FileNotFoundError):
        builder.build()


def test_build_create_config_kwargs(tmp_path):
    """Test _build_create_config_kwargs method."""
    input_onnx_path = tmp_path / "input_model.onnx"
    output_path = tmp_path / "output_model.plan"
    builder = TensorRTBuilder(input_onnx_path=input_onnx_path, output_path=output_path)

    # Mock trt.MemoryPoolType.WORKSPACE
    with patch("aitune.torch.backend.tensorrt.tensorrt_builder.trt") as mock_trt:
        mock_trt.MemoryPoolType.WORKSPACE = "WORKSPACE"

        # Create mock profiles
        profiles = [MagicMock()]

        # Test with basic parameters
        kwargs = builder._build_create_config_kwargs(
            max_workspace_size=1 << 30,
            optimization_level=None,
            compatibility_level=None,
            profiles=profiles,
            timing_cache=None,
        )

        # Verify kwargs
        assert kwargs["memory_pool_limits"] == {"WORKSPACE": 1 << 30}
        assert kwargs["profiles"] == profiles

        # Test with optimization level
        kwargs = builder._build_create_config_kwargs(
            max_workspace_size=1 << 30,
            optimization_level=5,
            compatibility_level=None,
            profiles=profiles,
            timing_cache=None,
        )

        assert kwargs["builder_optimization_level"] == 5


@pytest.mark.skipif(
    not is_cuda_available(),
    reason="CUDA is not available",
)
def test_tensorrt_builder_integration_linear(tmp_path):
    """Integration test for TensorRTBuilder with actual TensorRT.

    This test uses the toy_linear.onnx model from the toy_models package
    to test building and running a TensorRT engine from an ONNX model.
    """
    # Get path to the toy ONNX model
    onnx_path = ToyOnnxModel(is_linear=True).path
    assert onnx_path.exists(), f"ONNX model not found at {onnx_path}"

    # Create a TensorRTBuilder instance with profiles
    # Add profiles with min/opt/max shapes for the dynamic inputs
    min_shapes = {"input": (1, 256)}
    opt_shapes = {"input": (1, 256)}
    max_shapes = {"input": (8, 256)}

    # Build engine from ONNX model
    engine_path = tmp_path / "tensorrt" / "toy_linear.plan"
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    builder = TensorRTBuilder(
        input_onnx_path=onnx_path,
        output_path=engine_path,
        min_shapes=min_shapes,
        opt_shapes=opt_shapes,
        max_shapes=max_shapes,
    )
    returned_path = builder.build()

    # Verify engine file was created
    assert returned_path.exists(), f"Engine file not created at {returned_path}"

    # Load engine and create execution context
    runtime = TensorRTRuntime()
    engine_bytes = runtime.load_engine(engine_path)
    context, bindings, input_names, output_names, engine_info = runtime.create_execution_context(engine_bytes)

    # Create sample input data (ToyTorchModel uses input shape (batch_size, 256))
    batch_size = 1
    input_data = torch.rand(batch_size, 256, dtype=torch.float32)

    # Verify input and output names
    assert len(input_names) > 0, "No input tensors found in the engine"
    assert len(output_names) > 0, "No output tensors found in the engine"
    assert len(input_names) == len(engine_info.input_names), "Input names do not match"
    assert len(output_names) == len(engine_info.output_names), "Output names do not match"

    # Set input tensor
    input_tensor_name = input_names[0]
    bindings[input_tensor_name] = input_data.contiguous().data_ptr()

    # Set input dimensions for the execution context
    context.set_input_shape(input_tensor_name, tuple(input_data.shape))

    # Allocate output tensor
    output_tensor_name = output_names[0]
    output_shape = context.get_tensor_shape(output_tensor_name)

    # Fix dynamic shape: replace -1 with the actual batch size
    if output_shape[0] == -1:
        output_shape[0] = batch_size

    output_tensor = torch.empty(tuple(output_shape), dtype=torch.float32)
    bindings[output_tensor_name] = output_tensor.data_ptr()


@pytest.mark.skipif(
    not is_cuda_available(),
    reason="CUDA is not available",
)
def test_tensorrt_builder_integration_conv(tmp_path):
    """Integration test for TensorRTBuilder with actual TensorRT.

    This test uses the toy_conv.onnx model from the toy_models package
    to test building and running a TensorRT engine from an ONNX model.
    """
    try:
        # Get path to the toy ONNX model
        onnx_path = ToyOnnxModel(is_linear=False).path
        assert onnx_path.exists(), f"ONNX model not found at {onnx_path}"
    except FileNotFoundError:
        pytest.skip("Convolutional toy ONNX model not available")

    # Create a TensorRTBuilder instance with profiles
    # Add profiles with min/opt/max shapes for the dynamic inputs
    min_shapes = {"input": (1, 1, 129, 129)}
    opt_shapes = {"input": (1, 1, 129, 129)}
    max_shapes = {"input": (8, 1, 129, 129)}

    # Build engine from ONNX model
    engine_path = tmp_path / "tensorrt" / "toy_conv.plan"
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    builder = TensorRTBuilder(
        input_onnx_path=onnx_path,
        output_path=engine_path,
        min_shapes=min_shapes,
        opt_shapes=opt_shapes,
        max_shapes=max_shapes,
    )
    returned_path = builder.build()

    # Verify engine file was created
    assert returned_path.exists(), f"Engine file not created at {returned_path}"

    # Load engine and create execution context
    runtime = TensorRTRuntime()
    engine_bytes = runtime.load_engine(engine_path)
    context, bindings, input_names, output_names, engine_info = runtime.create_execution_context(engine_bytes)

    assert engine_info is not None
    assert engine_info.input_names is not None
    assert engine_info.output_names is not None
    assert len(engine_info.input_names) == len(input_names)
    assert len(engine_info.output_names) == len(output_names)

    # Create sample input data (ToyTorchModel uses input shape (batch_size, 1, 129, 129))
    batch_size = 1
    input_data = torch.rand(batch_size, 1, 129, 129, dtype=torch.float32)

    # Verify input and output names
    assert len(input_names) > 0, "No input tensors found in the engine"
    assert len(output_names) > 0, "No output tensors found in the engine"

    # Set input tensor
    input_tensor_name = input_names[0]
    bindings[input_tensor_name] = input_data.contiguous().data_ptr()

    # Set input dimensions for the execution context
    context.set_input_shape(input_tensor_name, tuple(input_data.shape))

    # Allocate output tensor
    output_tensor_name = output_names[0]
    output_shape = context.get_tensor_shape(output_tensor_name)

    # Fix dynamic shape: replace -1 with the actual batch size
    if output_shape[0] == -1:
        output_shape[0] = batch_size

    output_tensor = torch.empty(tuple(output_shape), dtype=torch.float32)
    bindings[output_tensor_name] = output_tensor.data_ptr()


def test_build_with_profile_objects(mock_polygraphy, tmp_path):
    """Test build method with TensorRTProfile objects."""
    # Create mock ONNX path
    onnx_path = tmp_path / "test_model.onnx"
    with open(onnx_path, "w") as f:
        f.write("mock onnx content")

    # Create mock profile objects
    mock_profile1 = MagicMock()
    mock_profile1.profile = "profile1"
    mock_profile2 = MagicMock()
    mock_profile2.profile = "profile2"
    profiles = [mock_profile1, mock_profile2]

    # Create builder with profiles
    engine_path = tmp_path / "tensorrt" / "test_model.plan"
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    builder = TensorRTBuilder(input_onnx_path=onnx_path, output_path=engine_path, profiles=profiles)

    # Build engine
    returned_path = builder.build()

    # Verify build interactions
    # Check that the profiles were passed to CreateConfig
    profiles_arg = mock_polygraphy["config"].call_args[1]["profiles"]
    assert len(profiles_arg) == 2
    assert profiles_arg[0] == "profile1"
    assert profiles_arg[1] == "profile2"

    # Verify other interactions
    mock_polygraphy["network"].assert_called_once_with(str(onnx_path), strongly_typed=True)
    mock_polygraphy["engine"].assert_called_once_with(
        network=mock_polygraphy["network"].return_value, config=mock_polygraphy["config"].return_value
    )
    mock_polygraphy["save"].assert_called_once_with(
        engine=mock_polygraphy["engine"].return_value, path=str(engine_path)
    )

    # Verify returned path
    assert returned_path == engine_path


def test_build_with_multiple_profiles(mock_polygraphy, tmp_path, mocker):
    """Test building with multiple profile objects."""
    # Create mock ONNX path
    onnx_path = tmp_path / "test_model.onnx"
    with open(onnx_path, "w") as f:
        f.write("mock onnx content")

    # Create mock profile objects that return actual profile values
    # that can be passed to CreateConfig
    profile1 = mocker.MagicMock()
    profile1.profile = "profile1"

    profile2 = mocker.MagicMock()
    profile2.profile = "profile2"

    # Create builder with profiles
    engine_path = tmp_path / "tensorrt" / "test_model.plan"
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    builder = TensorRTBuilder(input_onnx_path=onnx_path, output_path=engine_path, profiles=[profile1, profile2])

    # Build engine
    returned_path = builder.build()

    # Verify profiles were passed to config creation
    config_call_kwargs = mock_polygraphy["config"].call_args[1]
    assert "profiles" in config_call_kwargs
    assert len(config_call_kwargs["profiles"]) == 2
    assert config_call_kwargs["profiles"] == ["profile1", "profile2"]

    # Verify build interactions
    mock_polygraphy["network"].assert_called_once_with(str(onnx_path), strongly_typed=True)
    mock_polygraphy["engine"].assert_called_once_with(
        network=mock_polygraphy["network"].return_value, config=mock_polygraphy["config"].return_value
    )
    mock_polygraphy["save"].assert_called_once_with(
        engine=mock_polygraphy["engine"].return_value, path=str(engine_path)
    )

    # Verify returned path
    assert returned_path == engine_path


def test_build_with_invalid_shapes(mock_polygraphy, tmp_path, mocker):
    """Test building with invalid shape configurations."""
    # Create mock ONNX path
    onnx_path = tmp_path / "test_model.onnx"
    with open(onnx_path, "w") as f:
        f.write("mock onnx content")

    # Create shapes with inconsistent dimensions
    mock_profile = mocker.patch("aitune.torch.backend.tensorrt.tensorrt_builder.Profile")
    # Create a mock instance
    mock_profile_instance = mocker.MagicMock()
    mock_profile.return_value = mock_profile_instance

    # Make add() raise a ValueError when called with mismatched dimensions
    mock_profile_instance.add.side_effect = ValueError("Dimensions don't match")

    # Create builder with invalid shapes
    min_shapes = {"input": (1, 3, 224, 224)}
    opt_shapes = {"input": (4, 3, 224, 224)}
    # Max shape has wrong dimensions
    max_shapes = {"input": (8, 3, 224, 224, 1)}  # Extra dimension

    engine_path = tmp_path / "tensorrt" / "test_model.plan"
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    builder = TensorRTBuilder(
        input_onnx_path=onnx_path,
        output_path=engine_path,
        min_shapes=min_shapes,
        opt_shapes=opt_shapes,
        max_shapes=max_shapes,
    )

    # Building should fail because of the shape mismatch
    with pytest.raises(ValueError):
        builder.build()

    # Verify Profile was created
    mock_profile.assert_called_once()

    # Verify attempt to add shape was made
    mock_profile_instance.add.assert_called_once_with(
        name="input", min=(1, 3, 224, 224), opt=(4, 3, 224, 224), max=(8, 3, 224, 224, 1)
    )
