# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PyTorch module utilities."""

import gc
from collections import UserDict
from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from aitune.torch import register_for_jit_tuning
from aitune.torch.backend.backend import BuildMode
from aitune.torch.jit.patcher import jit_reset
from aitune.torch.module.onnx_module import OnnxModule
from aitune.torch.utils.module import (
    count_parameters,
    format_num_parameters,
    is_distributed_module,
    is_externally_managed_module,
    move_module_to_device,
    move_tensors_to_device,
    offload,
    offload_after_tuning,
    register_externally_managed_module,
    unregister_externally_managed_module,
)
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


def test_ordinary_module_is_not_distributed():
    assert not is_distributed_module(nn.Linear(2, 2))


def test_distributed_module_uses_enabled_integration_detector(mocker):
    detector = mocker.patch("aitune.torch.utils.module.is_integration_distributed_module", return_value=True)
    module = nn.Linear(2, 2)

    assert is_distributed_module(module)
    detector.assert_called_once_with(module)


def test_distributed_module_detects_dtensor_parameters(mocker):
    module = nn.Linear(2, 2)
    mocker.patch("aitune.torch.utils.module.DTensor", nn.Parameter)

    assert is_distributed_module(module)


def test_distributed_module_detects_distributed_children():
    distributed_identity = type("DistributedIdentity", (nn.Identity,), {"__module__": "torch.distributed.test"})
    module = nn.Sequential(distributed_identity())

    assert is_distributed_module(module)


def test_distributed_module_detects_distributed_data_parallel():
    module = object.__new__(DistributedDataParallel)

    assert is_distributed_module(module)


def test_move_module_moves_ordinary_module():
    module = nn.Linear(2, 2)

    move_module_to_device(module, torch.device("meta"))

    assert next(module.parameters()).device.type == "meta"


def test_move_module_preserves_distributed_module_placement(mocker):
    module = nn.Linear(2, 2)
    mocker.patch("aitune.torch.utils.module.is_distributed_module", return_value=True)

    move_module_to_device(module, torch.device("meta"))

    assert next(module.parameters()).device.type == "cpu"


@pytest.fixture
def parent_with_external_child():
    parent = nn.Module()
    parent.register_parameter("weight", nn.Parameter(torch.ones(2, 2)))
    parent.child = nn.Module()
    parent.child.layer = nn.Linear(2, 2)
    parent.sibling = nn.Linear(2, 2)
    external_root = parent.child
    register_externally_managed_module(external_root)
    try:
        yield parent
    finally:
        unregister_externally_managed_module(external_root)


@pytest.mark.parametrize("operation", [move_module_to_device, offload])
def test_parent_bulk_placement_rejects_external_subtree_before_any_mutation(parent_with_external_child, operation):
    parent = parent_with_external_child
    original_parameters = tuple(parent.parameters())
    original_values = [parameter.detach().clone() for parameter in original_parameters]
    assert not is_externally_managed_module(parent)

    with pytest.raises(RuntimeError, match="contains an externally managed subtree.*independent ordinary submodules"):
        operation(parent, "meta")

    for current, original, values in zip(parent.parameters(), original_parameters, original_values, strict=True):
        assert current is original
        assert current.device.type == "cpu"
        torch.testing.assert_close(current, values)


@pytest.mark.parametrize("operation", [move_module_to_device, offload])
@pytest.mark.parametrize("descendant", [False, True])
def test_external_root_and_descendant_keep_existing_noop(parent_with_external_child, operation, descendant):
    target = parent_with_external_child.child
    if descendant:
        target = target.layer
    assert is_externally_managed_module(target)

    operation(target, "meta")

    assert all(parameter.device.type == "cpu" for parameter in parent_with_external_child.parameters())


@pytest.mark.parametrize("operation", [move_module_to_device, offload])
def test_independent_sibling_and_parent_copy_remain_movable(parent_with_external_child, operation):
    parent = parent_with_external_child
    copied_parent = deepcopy(parent)

    operation(parent.sibling, "meta")
    operation(copied_parent, "meta")

    assert all(parameter.device.type == "meta" for parameter in parent.sibling.parameters())
    assert all(parameter.device.type == "meta" for parameter in copied_parent.parameters())
    assert all(parameter.device.type == "cpu" for parameter in parent.child.parameters())
    assert parent.weight.device.type == "cpu"
    assert not is_externally_managed_module(copied_parent.child)


@pytest.mark.parametrize("operation", [move_module_to_device, offload])
@pytest.mark.parametrize("cleanup", ["unregister", "reset"])
def test_external_placement_cleanup_restores_parent_movement(parent_with_external_child, operation, cleanup):
    parent = parent_with_external_child
    if cleanup == "reset":
        register_for_jit_tuning([parent.child], device_management="external")
        jit_reset()
    else:
        unregister_externally_managed_module(parent.child)

    operation(parent, "meta")

    assert all(parameter.device.type == "meta" for parameter in parent.parameters())


def test_parent_offload_rejection_preserves_onnx_sessions_and_skips_cleanup(
    parent_with_external_child, tmp_path, mocker
):
    parent = parent_with_external_child
    parent.onnx = OnnxModule(tmp_path / "unused.onnx")
    offload_onnx = mocker.patch.object(parent.onnx, "offload")
    cleanup = mocker.patch("aitune.torch.utils.module.cleanup_memory")

    with pytest.raises(RuntimeError, match="Cannot offload"):
        offload(parent, "meta")

    offload_onnx.assert_not_called()
    cleanup.assert_not_called()
    assert all(parameter.device.type == "cpu" for parameter in parent.parameters())


@pytest.mark.parametrize("operation", [move_module_to_device, offload])
def test_distributed_parent_preserves_existing_noop_with_external_child(parent_with_external_child, operation):
    parent = parent_with_external_child
    distributed_identity = type("DistributedIdentity", (nn.Identity,), {"__module__": "torch.distributed.test"})
    parent.distributed = distributed_identity()

    operation(parent, "meta")

    assert all(parameter.device.type == "cpu" for parameter in parent.parameters())


@pytest.mark.parametrize("operation", [move_module_to_device, offload])
def test_parent_rejection_preserves_shared_parameter_identity(parent_with_external_child, operation):
    parent = parent_with_external_child
    parent.sibling.weight = parent.child.layer.weight
    shared_weight = parent.sibling.weight

    with pytest.raises(RuntimeError, match="externally managed subtree"):
        operation(parent, "meta")

    assert parent.sibling.weight is shared_weight
    assert parent.child.layer.weight is shared_weight
    assert shared_weight.device.type == "cpu"


@pytest.mark.parametrize("operation", [move_module_to_device, offload])
def test_dynamically_shared_external_descendant_prevents_parent_movement(operation):
    external_root = nn.Module()
    external_root.layer = nn.Linear(2, 2)
    parent = nn.Module()
    parent.ordinary = nn.Linear(2, 2)
    register_externally_managed_module(external_root)
    try:
        parent.shared = external_root.layer
        assert not is_externally_managed_module(parent)

        with pytest.raises(RuntimeError, match="externally managed subtree"):
            operation(parent, "meta")

        assert external_root.layer.weight.device.type == "cpu"
        assert parent.ordinary.weight.device.type == "cpu"
    finally:
        unregister_externally_managed_module(external_root)


def test_dead_external_root_does_not_prevent_former_descendant_movement():
    external_root = nn.Module()
    external_root.layer = nn.Linear(2, 2)
    former_descendant = external_root.layer
    register_externally_managed_module(external_root)
    del external_root
    gc.collect()

    move_module_to_device(former_descendant, "meta")

    assert former_descendant.weight.device.type == "meta"


def test_move_tensors_preserves_dtensor_placement(mocker):
    tensor = torch.ones(2)
    mocker.patch("aitune.torch.utils.module.DTensor", torch.Tensor)

    result = move_tensors_to_device(({"tensor": tensor},), torch.device("meta"))

    assert result[0]["tensor"] is tensor


def test_move_tensors_moves_nested_ordinary_tensors():
    result = move_tensors_to_device(({"tensor": torch.ones(2)},), torch.device("meta"))

    assert result[0]["tensor"].device.type == "meta"


def test_move_tensors_moves_tensors_in_user_dict():
    value = UserDict({"tensor": torch.ones(2)})

    result = move_tensors_to_device(value, torch.device("meta"))

    assert result is value
    assert result["tensor"].device.type == "meta"


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


def test_offload_after_tuning_offloads_aot_module(mocker):
    model = mocker.Mock(spec=nn.Module)
    backend = mocker.Mock(build_mode=BuildMode.AHEAD_OF_TIME)
    offload_mock = mocker.patch("aitune.torch.utils.module.offload")

    offload_after_tuning(model, [backend], device="cpu")

    offload_mock.assert_called_once_with(model, device="cpu")


def test_offload_after_tuning_skips_jit_module(mocker):
    model = mocker.Mock(spec=nn.Module)
    backend = mocker.Mock(build_mode=BuildMode.JUST_IN_TIME)
    offload_mock = mocker.patch("aitune.torch.utils.module.offload")

    offload_after_tuning(model, [backend], device="meta")

    offload_mock.assert_not_called()


def test_offload_preserves_distributed_module_placement(mocker):
    model = mocker.Mock(spec=nn.Module)
    mocker.patch("aitune.torch.utils.module.is_distributed_module", return_value=True)
    cleanup = mocker.patch("aitune.torch.utils.module.cleanup_memory")

    offload(model, device="meta")

    model.to.assert_not_called()
    cleanup.assert_not_called()


@requires_cuda
def test_offload_to_meta_frees_gpu_memory(device):
    """Test that offload_to_meta frees GPU memory by replacing tensors with meta.

    Replacing parameter tensors with meta tensors (which have no data) should
    free the GPU memory that was used by the original parameter tensors.
    """
    # Clear CUDA cache first to get accurate measurements
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    gpu_memory_baseline = torch.cuda.memory_allocated(device)

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
    assert memory_allocated_before > gpu_memory_baseline, "Model should allocate some GPU memory"

    # Offload weights to meta (replace tensors)
    offload(model, device="meta")

    # Force memory deallocation
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Verify memory is freed (should be much less)
    memory_allocated_after = torch.cuda.memory_allocated(device)
    memory_freed = memory_allocated_before - memory_allocated_after

    # At least 50% of memory should be freed (conservative threshold)
    # Using 50% instead of 80% to account for:
    # - Small models where overhead is proportionally larger
    # - CUDA context and other allocations
    # - Potential fragmentation
    threshold = (memory_allocated_before - gpu_memory_baseline) * 0.5
    assert memory_freed > threshold, (
        f"Expected at least {threshold / 1e6:.2f} MB freed, "
        f"but only {memory_freed / 1e6:.2f} MB was freed. "
        f"Before: {memory_allocated_before / 1e6:.2f} MB, "
        f"After: {memory_allocated_after / 1e6:.2f} MB "
        f"Baseline: {gpu_memory_baseline / 1e6:.2f} MB"
    )


@requires_cuda
def test_offload_gpu_to_cpu_to_meta(device):
    """Test complete offload sequence: GPU -> CPU -> meta.

    This test verifies that:
    1. Loading to GPU allocates GPU memory
    2. Offloading to CPU frees GPU memory
    3. Offloading to meta frees CPU memory
    """
    # Clear CUDA cache first
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    gpu_memory_baseline = torch.cuda.memory_allocated(device)

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
    assert gpu_memory_after_load > gpu_memory_baseline, "Model should allocate GPU memory"

    # Step 2: Offload to CPU (should free GPU memory)
    offload(model, device="cpu")
    torch.cuda.synchronize()

    gpu_memory_after_cpu_offload = torch.cuda.memory_allocated(device)

    # GPU memory should be significantly reduced (at least 40%)
    gpu_memory_freed = gpu_memory_after_load - gpu_memory_after_cpu_offload
    assert gpu_memory_freed > (gpu_memory_after_load - gpu_memory_baseline) * 0.5, (
        f"Expected at least 50% of GPU memory freed after CPU offload. "
        f"Before: {gpu_memory_after_load / 1e6:.2f} MB, "
        f"After: {gpu_memory_after_cpu_offload / 1e6:.2f} MB, "
        f"Freed: {gpu_memory_freed / 1e6:.2f} MB "
        f"Baseline: {gpu_memory_baseline / 1e6:.2f} MB"
    )

    # Verify model is on CPU
    assert next(model.parameters()).device.type == "cpu"

    # Step 3: Offload to meta (should free CPU memory and move to meta)
    offload(model, device="meta")

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

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
