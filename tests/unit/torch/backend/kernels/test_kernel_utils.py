# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for kernel utilities."""

import torch
import torch.nn.functional as F  # noqa: N812

import aitune.torch.backend.kernels.kernel_utils as kernel_utils_module
from aitune.torch.backend.kernels.kernel_utils import KernelUtils
from tests.utilities.helpers import requires_cuda


def validate_function_with_tensors(real_output, test_output, *, atol=1e-4, rtol=1e-4):
    def real_function(*_args, **_kwargs):
        return real_output

    def test_function(*_args, **_kwargs):
        return test_output

    samples = [((), {})]
    return KernelUtils().validate_function(real_function, test_function, samples, atol=atol, rtol=rtol)


def test_validate_function_accepts_matching_tensor_output():
    """Matching tensor outputs validate successfully."""
    samples = [((torch.randn(2, 3), torch.randn(4, 3), torch.randn(4)), {})]

    errors = KernelUtils().validate_function(F.linear, F.linear, samples)

    assert not errors


def test_validate_function_calls_functions_without_grad():
    grad_enabled = []

    def function(value):
        grad_enabled.append(torch.is_grad_enabled())
        return value

    sample = ((torch.randn(2, 3),), {})

    errors = KernelUtils().validate_function(function, function, [sample])

    assert not errors
    assert grad_enabled == [False, False]


def test_benchmark_function_calls_function_without_grad(monkeypatch):
    grad_enabled = []

    def function(value):
        grad_enabled.append(torch.is_grad_enabled())
        return value

    def do_bench(test_fn, **_kwargs):
        test_fn()
        return 1.0

    monkeypatch.setattr(kernel_utils_module.triton.testing, "do_bench", do_bench)

    result = KernelUtils().benchmark_function(function, [((torch.randn(2, 3),), {})])

    assert result == 1.0
    assert grad_enabled == [False]


def test_validate_function_reports_shape_mismatch():
    """Tensor shape changes are rejected before tolerance checks."""
    x = torch.randn(2, 3)
    errors = validate_function_with_tensors(x, x.reshape(3, 2))

    assert len(errors) == 1
    assert errors[0] == (
        "Tensor metadata mismatch, expected: output shape=[2, 3] dtype=torch.float32, "
        "got: output shape=[3, 2] dtype=torch.float32"
    )


def test_validate_function_reports_tensor_structure_mismatch():
    """Different tensor output locations are rejected."""
    x = torch.randn(2, 3)
    errors = validate_function_with_tensors((x,), {"x": x})

    assert len(errors) == 1
    assert errors[0] == (
        "Tensor metadata mismatch, expected: output[0] shape=[2, 3] dtype=torch.float32, "
        'got: output["x"] shape=[2, 3] dtype=torch.float32'
    )


def test_validate_function_reports_dtype_mismatch():
    """Tensor dtype changes are rejected before tolerance checks."""
    x = torch.randn(2, 3)
    errors = validate_function_with_tensors(x, x.to(torch.int64))

    assert len(errors) == 1
    assert (
        errors[0]
        == "The values for attribute 'dtype' do not match: torch.float32 != torch.int64.\nAffected tensor: output"
    )


def test_validate_function_reports_nan_output():
    """NaN values in tensor outputs are rejected."""
    x = torch.ones(2, 3)
    errors = validate_function_with_tensors(x, torch.full_like(x, float("nan")))

    assert len(errors) == 1
    assert "Tensor-likes are not close!" in errors[0]


def test_validate_function_reports_inf_output():
    """Infinite values in tensor outputs are rejected."""
    x = torch.ones(2, 3)
    errors = validate_function_with_tensors(x, torch.full_like(x, float("inf")))

    assert len(errors) == 1
    assert "Tensor-likes are not close!" in errors[0]


def test_validate_function_reports_tolerance_mismatch_in_nested_tensor():
    """Nested tensor values must match within the provided tolerance."""
    x = torch.ones(2, 3)
    errors = validate_function_with_tensors((x, x + 1), (x, x + 2), atol=1e-6, rtol=1e-6)

    assert len(errors) == 1
    assert "Tensor-likes are not close!" in errors[0]


def test_validate_function_compares_non_tensor_outputs_when_no_tensors_are_returned():
    """Non-tensor-only outputs fall back to equality comparison."""
    errors = validate_function_with_tensors(2, 3)

    assert len(errors) == 1
    assert "Other data mismatch, expected" in errors[0]


def test_validate_function_reports_non_tensor_mismatch_in_mixed_output():
    """Non-tensor leaves in mixed outputs must also match."""
    x = torch.ones(2, 3)
    errors = validate_function_with_tensors((x, "real"), (x, "new"))

    assert len(errors) == 1
    assert "Other data mismatch, expected" in errors[0]


@requires_cuda
def test_benchmark_function():
    """Benchmark the function with given sample."""
    x = torch.randn(2, 3, device="cuda")
    weight = torch.randn(3, 3, device="cuda")
    sample = ((x,), {"weight": weight})  # check args and kwargs at the same time
    samples = [sample]

    for mode in ["min", "max", "mean", "median"]:
        result = KernelUtils().benchmark_function(F.linear, samples, return_mode=mode)
        assert result is not None
        assert isinstance(result, float)
        assert result > 0.0

    result = KernelUtils().benchmark_function(F.linear, samples, return_mode="all")
    assert result is not None
    assert isinstance(result, list)
    assert len(result) >= 1
