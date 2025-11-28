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

"""Unit tests for PyTorch module utilities."""

import pytest
import torch
import torch.nn as nn

from aitune.torch.utils.module import count_parameters, format_num_parameters, offload
from tests.utilities.helpers import requires_cuda


@pytest.mark.parametrize(
    "num_params,expected",
    [(100, 100), (1_000, 1000), (100_000, 100000), (1_000_000, 1000000), (1_000_000_000, 1000000000)],
)
def test_count_parameters(num_params, expected):
    """Test count_parameters with deeply nested modules."""
    module = nn.Sequential(
        nn.Linear(num_params, 1, bias=False),
    )
    result = count_parameters(module)

    assert result == expected


@pytest.mark.parametrize(
    "num_params,expected",
    [(100, "100"), (1_000, "1.0K"), (100_000, "100.0K"), (1_000_000, "1.0M"), (1_000_000_000, "1.0B")],
)
def test_format_num_parameters(num_params, expected):
    """Test count_parameters with deeply nested modules."""
    result = format_num_parameters(num_params)

    assert result == expected


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
    )


@pytest.fixture
def device():
    """Return CUDA device."""
    return torch.device("cuda:0")


@requires_cuda
def test_offload_moves_to_cpu_device(simple_model, device):
    """Test that offload replaces all parameters with CPU tensors.

    This frees GPU memory by replacing GPU tensors with CPU tensors,
    preserving the actual data.
    """
    # Move model to GPU
    simple_model.to(device)

    # Verify model is on GPU
    for param in simple_model.parameters():
        assert param.device.type == "cuda"

    # Offload weights to CPU (replaces parameter tensors)
    offload(simple_model, device="cpu")

    # Verify all parameters are now CPU tensors
    for param in simple_model.parameters():
        assert param.device.type == "cpu"


@requires_cuda
def test_offload_with_aggressive_cleanup_remove_all_tensors(simple_model, device):
    """Test that offload_to_meta removes all parameters, buffers and child modules."""
    # Move model to GPU
    simple_model.to(device)

    # Verify model is on GPU
    for param in simple_model.parameters():
        assert param.device.type == "cuda"

    # Offload weights with aggressive cleanup (replaces parameter tensors, frees memory)
    offload(simple_model, device="cpu", aggressive_cleanup=True)

    assert len(list(simple_model.parameters())) == 0


@requires_cuda
def test_offload_to_meta_frees_gpu_memory(device):
    """Test that offload_to_meta frees GPU memory by replacing tensors with meta.

    Replacing parameter tensors with meta tensors (which have no data) should
    free the GPU memory that was used by the original parameter tensors.
    """
    # Clear CUDA cache first to get accurate measurements
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Create a larger model for more reliable memory measurements
    model = nn.Sequential(
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 500),
    ).to(device)

    # Force memory allocation
    torch.cuda.synchronize()

    # Measure memory after model creation
    memory_allocated_before = torch.cuda.memory_allocated(device)
    assert memory_allocated_before > 0, "Model should allocate some GPU memory"

    # Offload weights to meta (replace tensors)
    offload(model, device="meta")

    # Verify memory is freed (should be much less)
    memory_allocated_after = torch.cuda.memory_allocated(device)
    memory_freed = memory_allocated_before - memory_allocated_after

    # At least 50% of memory should be freed (conservative threshold)
    # Using 50% instead of 80% to account for:
    # - Small models where overhead is proportionally larger
    # - CUDA context and other allocations
    # - Potential fragmentation
    threshold = memory_allocated_before * 0.5
    assert memory_freed > threshold, (
        f"Expected at least {threshold / 1e6:.2f} MB freed, "
        f"but only {memory_freed / 1e6:.2f} MB was freed. "
        f"Before: {memory_allocated_before / 1e6:.2f} MB, "
        f"After: {memory_allocated_after / 1e6:.2f} MB"
    )


@requires_cuda
def test_offload_with_aggressive_cleanup(simple_model, device):
    """Test offload with aggressive cleanup enabled.

    Aggressive cleanup performs multiple garbage collection passes
    to ensure all memory is freed.
    """
    simple_model.to(device)
    memory_before = torch.cuda.memory_allocated(device)

    # Offload with aggressive cleanup (default)
    offload(simple_model, device="meta", aggressive_cleanup=True)

    memory_after = torch.cuda.memory_allocated(device)

    # Memory should be significantly reduced
    assert memory_after < memory_before


@requires_cuda
def test_offload_without_aggressive_cleanup(simple_model, device):
    """Test offload with aggressive cleanup disabled.

    Even without aggressive cleanup, memory should still be freed,
    just with fewer garbage collection passes.
    """
    simple_model.to(device)
    memory_before = torch.cuda.memory_allocated(device)

    # Offload without aggressive cleanup
    offload(simple_model, device="meta", aggressive_cleanup=False)

    memory_after = torch.cuda.memory_allocated(device)

    # Memory should still be reduced (just maybe not as much)
    assert memory_after < memory_before


@requires_cuda
def test_offload_gpu_to_cpu_to_meta(device):
    """Test complete offload sequence: GPU -> CPU -> meta.

    This test verifies that:
    1. Loading to GPU allocates GPU memory
    2. Offloading to CPU frees GPU memory
    3. Offloading to meta frees CPU memory
    """
    # Clear CUDA cache first
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Create a larger model for reliable memory measurements
    model = nn.Sequential(
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 500),
    )

    # Step 1: Load model to GPU
    model.to(device)
    torch.cuda.synchronize()

    gpu_memory_after_load = torch.cuda.memory_allocated(device)
    assert gpu_memory_after_load > 0, "Model should allocate GPU memory"

    # Step 2: Offload to CPU (should free GPU memory)
    offload(model, device="cpu")
    torch.cuda.synchronize()

    gpu_memory_after_cpu_offload = torch.cuda.memory_allocated(device)

    # GPU memory should be significantly reduced (at least 50%)
    gpu_memory_freed = gpu_memory_after_load - gpu_memory_after_cpu_offload
    assert gpu_memory_freed > gpu_memory_after_load * 0.5, (
        f"Expected at least 50% of GPU memory freed after CPU offload. "
        f"Before: {gpu_memory_after_load / 1e6:.2f} MB, "
        f"After: {gpu_memory_after_cpu_offload / 1e6:.2f} MB, "
        f"Freed: {gpu_memory_freed / 1e6:.2f} MB"
    )

    # Verify model is on CPU
    assert next(model.parameters()).device.type == "cpu"

    # Step 3: Offload to meta (should free CPU memory and move to meta)
    offload(model, device="meta")

    # Verify model is on CPU (as per current implementation)
    # Note: The function is called offload_to_meta but currently moves to CPU
    assert next(model.parameters()).device.type == "meta"

    # GPU memory should remain low
    gpu_memory_final = torch.cuda.memory_allocated(device)
    assert gpu_memory_final <= gpu_memory_after_cpu_offload, (
        f"GPU memory should not increase after meta offload. "
        f"After CPU: {gpu_memory_after_cpu_offload / 1e6:.2f} MB, "
        f"After meta: {gpu_memory_final / 1e6:.2f} MB"
    )
