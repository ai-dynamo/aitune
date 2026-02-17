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
"""Tests for PyTorch-based TensorRT output allocator."""

from unittest.mock import MagicMock

import torch

from aitune.torch.backend.tensorrt.torch_output_allocator import TorchOutputAllocator
from tests.utilities.helpers import requires_cuda


# Mock TensorRT Dims class for testing
class MockDims:
    """Mock TensorRT Dims for testing."""

    def __init__(self, shape):
        self.shape = shape

    def __iter__(self):
        return iter(self.shape)

    def __getitem__(self, index):
        return self.shape[index]

    def __len__(self):
        return len(self.shape)


@requires_cuda
def test_torch_output_allocator_initialization():
    """Test TorchOutputAllocator initialization."""
    # Test initialization without engine info
    allocator = TorchOutputAllocator()
    assert allocator.outputs == {}
    assert allocator.output_shapes == {}
    assert allocator._engine_info is None

    # Test initialization with mock engine info
    mock_engine_info = MagicMock()
    mock_engine_info.output_dtypes = {"output_0": torch.float32}

    allocator_with_info = TorchOutputAllocator(engine_info=mock_engine_info)
    assert allocator_with_info._engine_info == mock_engine_info


@requires_cuda
def test_reallocate_output_basic():
    """Test basic output tensor reallocation."""
    allocator = TorchOutputAllocator()

    # Test allocating a float32 tensor
    tensor_name = "output_0"
    size = 1024 * 4  # 1024 float32 elements (4 bytes each)
    alignment = 16

    ptr = allocator.reallocate_output(tensor_name, 0, size, alignment)

    # Verify allocation succeeded
    assert ptr != 0
    assert tensor_name in allocator.outputs

    # Verify tensor properties
    tensor = allocator.outputs[tensor_name]
    assert tensor.is_cuda
    assert tensor.dtype == torch.float32
    assert tensor.numel() == 1024
    assert tensor.is_contiguous()
    assert tensor.data_ptr() == ptr


@requires_cuda
def test_reallocate_output_with_engine_info():
    """Test output tensor reallocation with engine info providing dtype."""
    # Create mock engine info with different dtypes
    mock_engine_info = MagicMock()
    mock_engine_info.output_dtypes = {
        "output_0": torch.float16,
        "output_1": torch.int32,
    }

    allocator = TorchOutputAllocator(engine_info=mock_engine_info)

    # Test float16 allocation
    ptr_fp16 = allocator.reallocate_output("output_0", 0, 1024 * 2, 16)  # 1024 fp16 elements
    assert ptr_fp16 != 0
    tensor_fp16 = allocator.outputs["output_0"]
    assert tensor_fp16.dtype == torch.float16
    assert tensor_fp16.numel() == 1024

    # Test int32 allocation
    ptr_int32 = allocator.reallocate_output("output_1", 0, 512 * 4, 16)  # 512 int32 elements
    assert ptr_int32 != 0
    tensor_int32 = allocator.outputs["output_1"]
    assert tensor_int32.dtype == torch.int32
    assert tensor_int32.numel() == 512


@requires_cuda
def test_notify_shape():
    """Test shape notification and tensor reshaping."""
    allocator = TorchOutputAllocator()

    # First allocate a tensor
    tensor_name = "output_0"
    size = 2 * 3 * 4 * 4  # 96 float32 elements for shape (2, 3, 4)
    ptr = allocator.reallocate_output(tensor_name, 0, size, 16)
    assert ptr != 0

    # Verify initial tensor is 1D
    original_tensor = allocator.outputs[tensor_name]
    assert original_tensor.dim() == 1
    assert original_tensor.numel() == 24  # 96 bytes / 4 bytes per float32

    # Notify shape - should reshape the tensor
    target_shape = (2, 3, 4)
    mock_dims = MockDims(target_shape)
    allocator.notify_shape(tensor_name, mock_dims)

    # Verify tensor was reshaped
    reshaped_tensor = allocator.outputs[tensor_name]
    assert reshaped_tensor.shape == target_shape
    assert reshaped_tensor.numel() == 24
    assert tensor_name in allocator.output_shapes
    assert allocator.output_shapes[tensor_name] == target_shape


@requires_cuda
def test_notify_shape_insufficient_elements():
    """Test shape notification when tensor has insufficient elements."""
    allocator = TorchOutputAllocator()

    # Allocate a small tensor
    tensor_name = "output_0"
    size = 8 * 4  # 8 float32 elements
    ptr = allocator.reallocate_output(tensor_name, 0, size, 16)
    assert ptr != 0

    # Try to reshape to larger shape than available elements
    target_shape = (4, 4, 4)  # Requires 64 elements, but we only have 8
    mock_dims = MockDims(target_shape)

    # This should not crash but should log a warning
    allocator.notify_shape(tensor_name, mock_dims)

    # Tensor should remain unchanged
    tensor = allocator.outputs[tensor_name]
    assert tensor.dim() == 1  # Still 1D
    assert tensor.numel() == 8


@requires_cuda
def test_outputs_property():
    """Test getting outputs from allocator."""
    allocator = TorchOutputAllocator()

    # Allocate some tensors
    allocator.reallocate_output("output_0", 0, 16, 16)
    allocator.reallocate_output("output_1", 0, 32, 16)

    # Add shapes
    allocator.notify_shape("output_0", MockDims((2, 2)))
    allocator.notify_shape("output_1", MockDims((2, 4)))

    # Get outputs
    outputs = allocator.outputs

    assert len(outputs) == 2
    assert "output_0" in outputs
    assert "output_1" in outputs
    assert outputs["output_0"].shape == (2, 2)
    assert outputs["output_1"].shape == (2, 4)

    # Verify it's a copy (modifying doesn't affect original)
    outputs.pop("output_0")
    assert "output_0" in allocator.outputs


@requires_cuda
def test_output_shapes_property():
    """Test getting output shapes from allocator."""
    allocator = TorchOutputAllocator()

    # Add some shapes
    allocator.notify_shape("output_0", MockDims((2, 3, 4)))
    allocator.notify_shape("output_1", MockDims((1, 10)))

    shapes = allocator.output_shapes

    assert len(shapes) == 2
    assert shapes["output_0"] == (2, 3, 4)
    assert shapes["output_1"] == (1, 10)

    # Verify the property returns the actual internal dictionary (not a copy)
    assert shapes is allocator._output_shapes


@requires_cuda
def test_clear():
    """Test clearing allocator."""
    allocator = TorchOutputAllocator()

    # Allocate some tensors and set shapes
    allocator.reallocate_output("output_0", 0, 16, 16)
    allocator.reallocate_output("output_1", 0, 32, 16)
    allocator.notify_shape("output_0", MockDims((2, 2)))
    allocator.notify_shape("output_1", MockDims((2, 4)))

    # Verify allocator has data
    assert len(allocator.outputs) == 2
    assert len(allocator.output_shapes) == 2

    # Clear and verify empty
    allocator.clear()
    assert len(allocator.outputs) == 0
    assert len(allocator.output_shapes) == 0


@requires_cuda
def test_allocation_failure_handling():
    """Test handling of allocation failures."""
    # Mock engine info that will cause issues during tensor creation
    mock_engine_info = MagicMock()
    # Set an invalid dtype that doesn't exist in our mapping
    mock_engine_info.output_dtypes = {"output_0": "invalid_dtype"}

    allocator = TorchOutputAllocator(engine_info=mock_engine_info)

    # Try to allocate - the invalid dtype will cause a ValueError to be caught
    # and return 0 instead of raising an exception
    ptr = allocator.reallocate_output("output_0", 0, 64, 16)
    assert ptr == 0, "Should return 0 on allocation failure"


@requires_cuda
def test_multiple_allocations_same_tensor():
    """Test multiple allocations for the same tensor name (reallocation)."""
    allocator = TorchOutputAllocator()

    tensor_name = "output_0"

    # First allocation
    ptr1 = allocator.reallocate_output(tensor_name, 0, 16, 16)
    assert ptr1 != 0
    tensor1 = allocator.outputs[tensor_name]

    # Second allocation (reallocation) - should replace the first
    ptr2 = allocator.reallocate_output(tensor_name, ptr1, 32, 16)
    assert ptr2 != 0
    tensor2 = allocator.outputs[tensor_name]

    # Should be a different tensor
    assert tensor2.data_ptr() == ptr2
    assert tensor2.numel() != tensor1.numel()  # Different sizes
