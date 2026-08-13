# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for module_function_kernel_profiler helpers."""

import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from aitune.torch.backend.kernels.module_function_kernel_profiler import (
    FunctionData,
    ModuleFunctionKernelProfiler,
    _nearest_parent_label,
    _ProfilingItem,
    _SummaryItem,
    get_tensor_size,
)
from tests.utilities.helpers import requires_cuda


class _MultipleLinearLayerModel(nn.Module):
    """Model combining Linear modules and direct functional linear calls."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(5, 5)
        self.linear2 = nn.Linear(5, 5)

    def forward(self, x):
        return (
            self.linear1(x)
            + self.linear2(x)
            + self.linear2(x)
            + F.linear(x, self.linear1.weight, self.linear1.bias)
            + F.linear(x, self.linear2.weight, self.linear2.bias)
            + F.linear(x, self.linear2.weight, self.linear2.bias)
        )

    def samples(self, batch_sizes: list[int], device: str = "cpu"):
        return [((torch.randn(batch_size, 5, device=device),), {}) for batch_size in batch_sizes]


class _FakeProfilerEvent:
    """Minimal stand-in for profiler events that expose cpu_parent and key."""

    __slots__ = ("cpu_parent", "key", "kernels")

    def __init__(self, *, cpu_parent=None, key="", kernels=None):
        self.cpu_parent = cpu_parent
        self.key = key
        self.kernels = kernels or []


class _FakeProfilerKernel:
    """Minimal stand-in for profiler kernel metadata."""

    name = "fake_kernel"
    duration = 1.0


class _CountingModuleFunctionKernelProfiler(ModuleFunctionKernelProfiler):
    """Profiler variant that records function data rewrites."""

    def __init__(self, function_names: set[str] | None = None):
        super().__init__(function_names=function_names)
        self.rewrite_calls: list[str] = []

    def _rewrite_func_data(self, func_name: str) -> FunctionData:
        self.rewrite_calls.append(func_name)
        return []


def _assert_linear_kernel_attribution(profiling_df, num_samples=1):
    """Assert module attribution while allowing multiple kernels per linear call."""
    assert profiling_df["function_name"].unique().tolist() == ["linear"]
    module_counts = profiling_df["module_name"].value_counts().to_dict()

    # Per sample, the model calls linear1 once, linear2 twice, and F.linear directly from the root module three times.
    # A functional call may launch one or more CUDA kernels, so the number of profiling rows is architecture-dependent,
    # but their module attribution must retain the root:linear1:linear2 ratio of 3:1:2.
    kernels_per_linear1_calls = module_counts["linear1"]
    assert kernels_per_linear1_calls >= num_samples
    assert module_counts == {
        "": 3 * kernels_per_linear1_calls,
        "linear1": kernels_per_linear1_calls,
        "linear2": 2 * kernels_per_linear1_calls,
    }


def test_get_functions_to_patch():
    profiler = ModuleFunctionKernelProfiler()
    functions = profiler.get_functions_to_patch()
    assert len(functions) > 1

    function_names = [func_name for func_name, _ in functions]
    assert "linear" in function_names

    profiler = ModuleFunctionKernelProfiler(function_names={"linear"})
    functions = profiler.get_functions_to_patch()
    function_names = [func_name for func_name, _ in functions]
    assert "linear" in function_names
    assert "relu" in function_names


def test_profile_requires_module_scope_for_non_module_function():
    profiler = ModuleFunctionKernelProfiler()

    with pytest.raises(ValueError, match="module must be provided"):
        profiler.profile(lambda: None)


def test_profile_rejects_negative_warmup_iterations():
    profiler = ModuleFunctionKernelProfiler()

    with pytest.raises(ValueError, match="warmup_iterations must be greater than or equal to 0"):
        profiler.profile(nn.Identity(), warmup_iterations=-1)


def test_nearest_parent_label_no_cpu_parent():
    """When the event has no cpu_parent chain, the helper returns None."""
    event = _FakeProfilerEvent(cpu_parent=None)
    assert _nearest_parent_label(event, "ait_nn_module:") is None


def test_nearest_parent_label_immediate_parent_matches():
    """The first cpu_parent with a matching prefix yields its label without the prefix."""
    parent = _FakeProfilerEvent(key="ait_nn_module:layer1", cpu_parent=None)
    event = _FakeProfilerEvent(cpu_parent=parent)
    assert _nearest_parent_label(event, "ait_nn_module:") == "layer1"


def test_nearest_parent_label_skips_non_matching_ancestors():
    """Non-matching parents are skipped until a matching ancestor is found."""
    root = _FakeProfilerEvent(key="ait_nn_module:deepest", cpu_parent=None)
    mid = _FakeProfilerEvent(key="other_op", cpu_parent=root)
    event = _FakeProfilerEvent(cpu_parent=mid)
    assert _nearest_parent_label(event, "ait_nn_module:") == "deepest"


def test_nearest_parent_label_prefers_closer_matching_parent():
    """The first matching ancestor along cpu_parent wins (nearest to the event)."""
    far = _FakeProfilerEvent(key="ait_nn_module:far", cpu_parent=None)
    near = _FakeProfilerEvent(key="ait_nn_module:near", cpu_parent=far)
    event = _FakeProfilerEvent(cpu_parent=near)
    assert _nearest_parent_label(event, "ait_nn_module:") == "near"


def test_nearest_parent_label_empty_suffix_when_key_equals_prefix():
    """removeprefix on a key that is exactly the prefix yields an empty string."""
    parent = _FakeProfilerEvent(key="pref:", cpu_parent=None)
    event = _FakeProfilerEvent(cpu_parent=parent)
    assert _nearest_parent_label(event, "pref:") == ""


def test_nearest_parent_label_missing_key_uses_empty_string():
    """Events without key fall back to '' and do not match the prefix."""

    class NoKeyParent:
        __slots__ = ("cpu_parent",)

        def __init__(self, cpu_parent=None):
            self.cpu_parent = cpu_parent

    root = _FakeProfilerEvent(key="ait_nn_module:ok", cpu_parent=None)
    mid = NoKeyParent(cpu_parent=root)
    event = _FakeProfilerEvent(cpu_parent=mid)
    assert _nearest_parent_label(event, "ait_nn_module:") == "ok"


def test_match_events_rewrites_function_data_once_per_function():
    """Repeated profiler events for one function should rewrite sample data once."""
    module_parent = _FakeProfilerEvent(key="ait_nn_module:layer")
    function_parent = _FakeProfilerEvent(key="ait_torch.nn.functional:linear", cpu_parent=module_parent)
    events = [
        _FakeProfilerEvent(key="aten::linear", cpu_parent=function_parent, kernels=[_FakeProfilerKernel()]),
        _FakeProfilerEvent(key="aten::linear", cpu_parent=function_parent, kernels=[_FakeProfilerKernel()]),
    ]
    profiler = _CountingModuleFunctionKernelProfiler()

    profiling_data, function_data = profiler._match_events_with_data(events, [("layer", nn.Identity())])  # type: ignore[arg-type]

    assert len(profiling_data) == 2
    assert function_data == {"linear": []}
    assert profiler.rewrite_calls == ["linear"]


def test_describe_results_includes_time_spent_pct_for_described_functions():
    profiler = ModuleFunctionKernelProfiler()
    profiling_df = pd.DataFrame([
        {"function_name": "linear", "module_name": "layer1", "kernel_us": 200.0},
        {"function_name": "linear", "module_name": "layer2", "kernel_us": 100.0},
        {"function_name": "relu", "module_name": "layer1", "kernel_us": 100.0},
        {"function_name": "softmax", "module_name": "layer1", "kernel_us": 600.0},
    ])
    function_data = {
        "linear": [(2, ((), {}))],
        "relu": [(1, ((), {}))],
        "softmax": [(1, ((), {}))],
    }

    df = profiler.describe_results(profiling_df, function_data, top_k=2)

    assert df["function_name"].tolist() == ["softmax", "linear"]
    assert df["time_spent_pct"].tolist() == [600.0 / 900.0 * 100.0, 300.0 / 900.0 * 100.0]


def test_describe_results_uses_nan_sample_metrics_for_functions_without_collected_data():
    profiler = ModuleFunctionKernelProfiler()
    profiling_df = pd.DataFrame([
        {"function_name": "linear", "module_name": "layer1", "kernel_us": 200.0},
        {"function_name": "relu", "module_name": "layer1", "kernel_us": 100.0},
    ])
    function_data = {"linear": [(1, ((torch.randn(1, 2), torch.randn(2, 2)), {}))]}

    df = profiler.describe_results(profiling_df, function_data, top_k=2)

    relu_row = df.loc[df["function_name"] == "relu"].iloc[0]
    assert pd.isna(relu_row["calls"])
    assert pd.isna(relu_row["num_distinct_samples"])
    assert pd.isna(relu_row["tensor_size_MB"])


def test_describe_results_uses_zero_sample_metrics_for_empty_collected_data():
    profiler = ModuleFunctionKernelProfiler()
    profiling_df = pd.DataFrame([
        {"function_name": "relu", "module_name": "layer1", "kernel_us": 100.0},
    ])

    df = profiler.describe_results(profiling_df, {"relu": []})

    row = df.iloc[0]
    assert row["calls"] == 0
    assert row["num_distinct_samples"] == 0
    assert row["tensor_size_MB"] == 0


@requires_cuda
def test_profile_and_describe_return_schema_stable_empty_dataframes():
    profiler = ModuleFunctionKernelProfiler()
    module = nn.Identity().to("cuda")
    data = [((torch.randn(1, device="cuda"),), {})]

    profiling_df, function_data = profiler.profile(module, data, warmup_iterations=0)
    summary_df = profiler.describe_results(profiling_df, function_data)

    assert profiling_df.empty
    assert tuple(profiling_df.columns) == tuple(_ProfilingItem.__annotations__)
    assert summary_df.empty
    assert tuple(summary_df.columns) == tuple(_SummaryItem.__annotations__)


@requires_cuda
@pytest.mark.parametrize(
    ("function_names", "expected_num_function_data_samples"),
    [
        (None, 1),
        ({"linear"}, 1),
        ({"relu"}, 0),
    ],
    ids=["all-functions", "linear-only", "relu-filtered-out"],
)
def test_profile_module_function_kernels(function_names, expected_num_function_data_samples):
    batch_size, dim = 2, 5
    # given
    profiler = ModuleFunctionKernelProfiler(function_names=function_names)
    net = _MultipleLinearLayerModel().to("cuda")
    original_forward = net.forward
    x = net.samples(batch_sizes=[batch_size], device="cuda")
    # when
    profiling_df, function_data = profiler.profile(net, x)
    # then

    # check profiling data
    _assert_linear_kernel_attribution(profiling_df)

    # check function data
    assert sum(len(samples) for samples in function_data.values()) == expected_num_function_data_samples

    summary_row = profiler.describe_results(profiling_df, function_data).iloc[0]
    assert net.forward == original_forward
    if not function_data:
        assert pd.isna(summary_row["calls"])
        assert pd.isna(summary_row["num_distinct_samples"])
        return

    assert summary_row["calls"] == 6
    assert summary_row["num_distinct_samples"] == 1
    _, sample = function_data["linear"][0]
    args, kwargs = sample
    assert len(args) == 3  # x, W, b
    assert args[0].shape == (batch_size, dim)
    assert args[1].shape == (dim, dim)
    assert args[2].shape == (dim,)
    assert kwargs == {}


@requires_cuda
def test_profile_module_function_kernels_multiple_samples():
    batch_size, dim = 2, 5
    # given
    profiler = ModuleFunctionKernelProfiler()
    net = _MultipleLinearLayerModel().to("cuda")
    data = net.samples(batch_sizes=[batch_size, 2 * batch_size], device="cuda")
    function_calls = 0

    def run_net(*args, **kwargs):
        nonlocal function_calls
        function_calls += 1
        return net(*args, **kwargs)

    # when
    profiling_df, function_data = profiler.profile(run_net, data, module=net, warmup_iterations=2)
    # then
    assert function_calls == 6  # two warmup iterations and one profiled iteration over two samples
    _assert_linear_kernel_attribution(profiling_df, num_samples=2)

    assert len(function_data["linear"]) == 2
    for counter, sample in function_data["linear"]:
        assert counter == 6

        # data
        args, kwargs = sample
        assert len(args) == 3  # x, W, b
        assert args[0].shape == (batch_size, dim) or args[0].shape == (2 * batch_size, dim)
        assert args[1].shape == (dim, dim)
        assert args[2].shape == (dim,)
        assert kwargs == {}


@requires_cuda
def test_profile_module_function_kernels_without_data_calls_function_once():
    batch_size, dim = 2, 5
    profiler = ModuleFunctionKernelProfiler()
    net = _MultipleLinearLayerModel().to("cuda")
    args, kwargs = net.samples(batch_sizes=[batch_size], device="cuda")[0]
    function_calls = 0

    def run_net():
        nonlocal function_calls
        function_calls += 1
        return net(*args, **kwargs)

    profiling_df, function_data = profiler.profile(
        run_net,
        module=net,
        warmup_iterations=2,
    )

    assert function_calls == 3  # two warmup calls and one profiled call
    _assert_linear_kernel_attribution(profiling_df)

    assert len(function_data) == 1
    counter, sample = function_data["linear"][0]
    assert counter == 6

    sample_args, sample_kwargs = sample
    assert len(sample_args) == 3
    assert sample_args[0].shape == (batch_size, dim)
    assert sample_args[1].shape == (dim, dim)
    assert sample_args[2].shape == (dim,)
    assert sample_kwargs == {}


@requires_cuda
def test_profile_and_describe():
    batch_size = 2
    # given
    profiler = ModuleFunctionKernelProfiler()
    net = _MultipleLinearLayerModel().to("cuda")
    data = net.samples(batch_sizes=[batch_size, 2 * batch_size], device="cuda")
    # when
    profiling_df, function_data = profiler.profile(net, data)
    df = profiler.describe_results(profiling_df, function_data)
    # then
    assert len(df) == 1
    row = df.iloc[0]
    assert row["function_name"] == "linear"
    assert row["num_distinct_samples"] == 2
    assert row["calls"] == 12
    assert row["num_modules"] == 3


def test_get_tensor_size_uses_logical_tensor_payload():
    args_storage = torch.randn(4, 4, dtype=torch.float32)
    kwargs_storage = torch.randn(4, 4, dtype=torch.float16)
    args = (args_storage[:1, :1],)
    kwargs = {"tensor": kwargs_storage[:2, :2]}

    assert get_tensor_size(args, kwargs) == 4 + 4 * 2  # 1x1x4 + 2x2x2 bytes
