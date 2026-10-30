# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for module_function_kernel_profiler helpers."""

import logging
import os
from unittest.mock import call

import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.profiler import ProfilerActivity

import aitune.torch.kernel_forge.module_function_kernel_profiler as kernel_profiler_module
from aitune.torch.distributed import DistributedContext, DistributedCoordinator
from aitune.torch.kernel_forge.module_function_kernel_profiler import (
    FunctionData,
    ModuleFunctionKernelProfiler,
    teardown_cuda_profiler,
)
from tests.utilities.helpers import requires_cuda


@pytest.fixture(autouse=True)
def reset_cuda_profiler_state(monkeypatch):
    monkeypatch.setattr(kernel_profiler_module, "_cuda_profiler_primed", False)
    monkeypatch.setattr(kernel_profiler_module, "_owns_teardown_cupti", False)
    monkeypatch.setattr(kernel_profiler_module, "_last_teardown_cupti_value", None)
    # Record the original value even when absent, so direct writes by the helpers are undone.
    monkeypatch.setenv("TEARDOWN_CUPTI", "0")
    monkeypatch.delenv("TEARDOWN_CUPTI", raising=False)


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


class _NestedFunctionCallsModel(nn.Module):
    """Model that uses nested function calls.

    F.multi_head_attention_forward calls F.linear and F.scaled_dot_product_attention.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.seq_len = 16
        self.batch_size = 2
        self.embed_dim = 64
        self.num_heads = 8
        self.device = "cuda"

        self.in_proj_weight = torch.randn(3 * self.embed_dim, self.embed_dim, device=self.device)
        self.in_proj_bias = torch.randn(3 * self.embed_dim, device=self.device)

        self.out_proj_weight = torch.randn(self.embed_dim, self.embed_dim, device=self.device)
        self.out_proj_bias = torch.randn(self.embed_dim, device=self.device)

    def forward(self, x):
        return F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.out_proj_weight,
            out_proj_bias=self.out_proj_bias,
            training=False,
            key_padding_mask=None,
            need_weights=False,  # if True, it uses scaled_dot_product_attention
            attn_mask=None,
            is_causal=False,
        )

    def samples(self):
        return [((torch.randn(self.seq_len, self.batch_size, self.embed_dim, device=self.device),), {})]


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


@requires_cuda
def test_profile_coordinates_distributed_execution(mocker):
    verify_equal = mocker.patch("aitune.torch.kernel_forge.module_function_kernel_profiler.coordinator.verify_equal")
    coordinate_failures = mocker.patch(
        "aitune.torch.kernel_forge.module_function_kernel_profiler.coordinator.raise_if_any_rank_fails"
    )
    profiler = kernel_profiler_module.ModuleFunctionKernelProfiler()
    module = nn.Identity().to("cuda")
    data = [((torch.randn(1, device="cuda"),), {})]

    profiler.profile(module, data, warmup_iterations=1)

    verify_equal.assert_called_once_with(1, "kernel profiler sample count")
    assert coordinate_failures.call_args_list == [
        call("Priming CUDA profiler"),
        call("Warming up module function kernel profiler"),
        call("Preparing module function kernel profiler"),
        call("Profiling module functions"),
    ]


@pytest.mark.parametrize("value", [None, "0", "1"])
def test_prime_cuda_profiler_respects_teardown_cupti_ownership(mocker, monkeypatch, value):
    if value is not None:
        monkeypatch.setenv("TEARDOWN_CUPTI", value)
    session_values = []
    profile = mocker.patch.object(kernel_profiler_module, "profile")
    profile.return_value.__enter__.side_effect = lambda: session_values.append(os.environ["TEARDOWN_CUPTI"])
    zeros = mocker.patch.object(kernel_profiler_module.torch, "zeros")
    synchronize = mocker.patch.object(kernel_profiler_module.torch.cuda, "synchronize")

    kernel_profiler_module.prime_cuda_profiler()

    assert session_values == ["0" if value is None else value]
    assert os.environ["TEARDOWN_CUPTI"] == session_values[0]
    assert kernel_profiler_module._owns_teardown_cupti is (value is None)
    assert kernel_profiler_module._cuda_profiler_primed is True
    profile.assert_called_once_with(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
    zeros.assert_called_once_with(1, device="cuda")
    synchronize.assert_called_once_with()

    if value is not None:
        kernel_profiler_module.teardown_cuda_profiler()
        assert os.environ["TEARDOWN_CUPTI"] == value
        assert session_values == [value]


def test_teardown_cuda_profiler_skips_when_not_primed(mocker):
    profile = mocker.patch.object(kernel_profiler_module, "profile")

    teardown_cuda_profiler()

    profile.assert_not_called()
    assert "TEARDOWN_CUPTI" not in os.environ


@pytest.mark.parametrize("owned", [False, True])
def test_teardown_cuda_profiler_sets_teardown_cupti_during_session(mocker, monkeypatch, owned):
    monkeypatch.setattr(kernel_profiler_module, "_cuda_profiler_primed", True)
    monkeypatch.setattr(kernel_profiler_module, "_owns_teardown_cupti", owned)
    if owned:
        monkeypatch.setenv("TEARDOWN_CUPTI", "0")
        monkeypatch.setattr(kernel_profiler_module, "_last_teardown_cupti_value", "0")
    events: list[tuple] = []

    class _RecordingProfile:
        def __init__(self, activities):
            events.append(("profile", activities))

        def __enter__(self):
            events.append(("enter", os.environ.get("TEARDOWN_CUPTI")))
            return self

        def __exit__(self, *exc_info):
            events.append(("exit", os.environ.get("TEARDOWN_CUPTI")))

    mocker.patch.object(kernel_profiler_module, "profile", _RecordingProfile)
    zeros = mocker.patch.object(kernel_profiler_module.torch, "zeros")
    zeros.return_value.relu_.side_effect = lambda: events.append(("kernel",))
    mocker.patch.object(
        kernel_profiler_module.torch.cuda, "synchronize", side_effect=lambda: events.append(("synchronize",))
    )

    kernel_profiler_module.teardown_cuda_profiler()

    assert events == [
        ("profile", [ProfilerActivity.CUDA]),
        ("enter", "1"),
        ("kernel",),
        ("exit", "1"),
        ("synchronize",),
    ]
    zeros.assert_called_once_with(1, device="cuda")
    # Kineto reads the variable again from its asynchronous teardown thread, so it must stay set.
    assert os.environ["TEARDOWN_CUPTI"] == "1"
    assert kernel_profiler_module._cuda_profiler_primed is False


def test_teardown_cuda_profiler_keeps_disabled_teardown_cupti(mocker, monkeypatch, caplog):
    monkeypatch.setenv("TEARDOWN_CUPTI", "0")
    monkeypatch.setattr(kernel_profiler_module, "_cuda_profiler_primed", True)
    profile = mocker.patch.object(kernel_profiler_module, "profile")

    with caplog.at_level(logging.WARNING, logger=kernel_profiler_module.__name__):
        kernel_profiler_module.teardown_cuda_profiler()

    profile.assert_not_called()
    assert os.environ["TEARDOWN_CUPTI"] == "0"
    assert kernel_profiler_module._cuda_profiler_primed is True
    assert any("TEARDOWN_CUPTI=0" in record.message for record in caplog.records)


def test_teardown_cuda_profiler_skips_session_for_user_enabled_teardown(mocker, monkeypatch, caplog):
    monkeypatch.setenv("TEARDOWN_CUPTI", "1")
    monkeypatch.setattr(kernel_profiler_module, "_cuda_profiler_primed", True)
    profile = mocker.patch.object(kernel_profiler_module, "profile")
    zeros = mocker.patch.object(kernel_profiler_module.torch, "zeros")
    synchronize = mocker.patch.object(kernel_profiler_module.torch.cuda, "synchronize")

    with caplog.at_level(logging.WARNING, logger=kernel_profiler_module.__name__):
        kernel_profiler_module.teardown_cuda_profiler()

    profile.assert_not_called()
    zeros.assert_not_called()
    synchronize.assert_not_called()
    assert os.environ["TEARDOWN_CUPTI"] == "1"
    assert kernel_profiler_module._cuda_profiler_primed is False
    assert kernel_profiler_module._owns_teardown_cupti is False
    assert not caplog.records


def test_profile_repeated_calls_toggle_owned_teardown_cupti(mocker):
    # Mocking CUDA availability must not make distributed context detection initialize CUDA.
    mocker.patch.object(kernel_profiler_module, "coordinator", DistributedCoordinator(DistributedContext()))
    mocker.patch.object(kernel_profiler_module.torch.cuda, "is_available", return_value=True)
    mocker.patch.object(kernel_profiler_module.torch.cuda, "empty_cache")
    mocker.patch.object(kernel_profiler_module.torch.cuda, "synchronize")
    mocker.patch.object(kernel_profiler_module.torch, "zeros")
    profile = mocker.patch.object(kernel_profiler_module, "profile")
    profile.return_value.__enter__.return_value.profiler.function_events = []
    session_values = []
    profile.side_effect = lambda **kwargs: session_values.append(os.environ["TEARDOWN_CUPTI"]) or profile.return_value
    profiler = ModuleFunctionKernelProfiler()
    data = [((torch.ones(1),), {})]

    profiler.profile(nn.Identity(), data, warmup_iterations=0)
    assert os.environ["TEARDOWN_CUPTI"] == "1"
    assert kernel_profiler_module._cuda_profiler_primed is False
    profiler.profile(nn.Identity(), data, warmup_iterations=0)

    # Each call primes and profiles with teardown disabled, then runs the throwaway teardown session.
    assert session_values == ["0", "0", "1", "0", "0", "1"]
    assert os.environ["TEARDOWN_CUPTI"] == "1"
    assert kernel_profiler_module._cuda_profiler_primed is False
    assert kernel_profiler_module._owns_teardown_cupti is True


@pytest.mark.parametrize(("value", "change_during_profiling"), [("0", False), ("1", True), ("external", True)])
def test_profiler_relinquishes_ownership_after_external_change(mocker, monkeypatch, value, change_during_profiling):
    mocker.patch.object(kernel_profiler_module, "profile")
    mocker.patch.object(kernel_profiler_module.torch, "zeros")
    mocker.patch.object(kernel_profiler_module.torch.cuda, "synchronize")
    kernel_profiler_module.prime_cuda_profiler()
    if not change_during_profiling:
        kernel_profiler_module.teardown_cuda_profiler()
    monkeypatch.setenv("TEARDOWN_CUPTI", value)
    if not change_during_profiling:
        kernel_profiler_module.prime_cuda_profiler()
    kernel_profiler_module.teardown_cuda_profiler()

    assert os.environ["TEARDOWN_CUPTI"] == value
    assert kernel_profiler_module._owns_teardown_cupti is False


@pytest.mark.parametrize("initial_value", [None, "0", "1"])
def test_teardown_owns_recreated_setting_and_next_priming_disables_teardown(mocker, monkeypatch, initial_value):
    if initial_value is not None:
        monkeypatch.setenv("TEARDOWN_CUPTI", initial_value)
    session_values = []
    profile = mocker.patch.object(kernel_profiler_module, "profile")
    profile.side_effect = lambda **kwargs: session_values.append(os.environ["TEARDOWN_CUPTI"]) or profile.return_value
    mocker.patch.object(kernel_profiler_module.torch, "zeros")
    mocker.patch.object(kernel_profiler_module.torch.cuda, "synchronize")

    kernel_profiler_module.prime_cuda_profiler()
    monkeypatch.delenv("TEARDOWN_CUPTI")
    kernel_profiler_module.teardown_cuda_profiler()

    assert os.environ["TEARDOWN_CUPTI"] == "1"
    assert kernel_profiler_module._last_teardown_cupti_value == "1"
    assert kernel_profiler_module._owns_teardown_cupti is True
    assert kernel_profiler_module._cuda_profiler_primed is False

    kernel_profiler_module.prime_cuda_profiler()
    kernel_profiler_module.teardown_cuda_profiler()

    assert session_values == [initial_value or "0", "1", "0", "1"]
    assert kernel_profiler_module._owns_teardown_cupti is True


def test_profile_tears_down_cuda_profiler_when_profiling_fails(mocker):
    mocker.patch.object(kernel_profiler_module.torch.cuda, "is_available", return_value=True)
    mocker.patch.object(kernel_profiler_module.coordinator, "verify_equal")
    empty_cache = mocker.patch.object(kernel_profiler_module.torch.cuda, "empty_cache")
    teardown = mocker.patch.object(kernel_profiler_module, "teardown_cuda_profiler")
    profile_kernels = mocker.patch.object(
        kernel_profiler_module.ModuleFunctionKernelProfiler,
        "_profile_kernels",
        side_effect=RuntimeError("profiling failed"),
    )

    with pytest.raises(RuntimeError, match="profiling failed"):
        kernel_profiler_module.ModuleFunctionKernelProfiler().profile(nn.Identity(), warmup_iterations=0)

    profile_kernels.assert_called_once()
    teardown.assert_called_once_with()
    empty_cache.assert_not_called()


def test_nearest_parent_label_no_cpu_parent():
    """When the event has no cpu_parent chain, the helper returns None."""
    event = _FakeProfilerEvent(cpu_parent=None)
    assert kernel_profiler_module._nearest_parent_label(event, "ait_nn_module:") is None


def test_nearest_parent_label_immediate_parent_matches():
    """The first cpu_parent with a matching prefix yields its label without the prefix."""
    parent = _FakeProfilerEvent(key="ait_nn_module:layer1", cpu_parent=None)
    event = _FakeProfilerEvent(cpu_parent=parent)
    assert kernel_profiler_module._nearest_parent_label(event, "ait_nn_module:") == "layer1"


def test_nearest_parent_label_skips_non_matching_ancestors():
    """Non-matching parents are skipped until a matching ancestor is found."""
    root = _FakeProfilerEvent(key="ait_nn_module:deepest", cpu_parent=None)
    mid = _FakeProfilerEvent(key="other_op", cpu_parent=root)
    event = _FakeProfilerEvent(cpu_parent=mid)
    assert kernel_profiler_module._nearest_parent_label(event, "ait_nn_module:") == "deepest"


def test_nearest_parent_label_prefers_closer_matching_parent():
    """The first matching ancestor along cpu_parent wins (nearest to the event)."""
    far = _FakeProfilerEvent(key="ait_nn_module:far", cpu_parent=None)
    near = _FakeProfilerEvent(key="ait_nn_module:near", cpu_parent=far)
    event = _FakeProfilerEvent(cpu_parent=near)
    assert kernel_profiler_module._nearest_parent_label(event, "ait_nn_module:") == "near"


def test_nearest_parent_label_empty_suffix_when_key_equals_prefix():
    """removeprefix on a key that is exactly the prefix yields an empty string."""
    parent = _FakeProfilerEvent(key="pref:", cpu_parent=None)
    event = _FakeProfilerEvent(cpu_parent=parent)
    assert kernel_profiler_module._nearest_parent_label(event, "pref:") == ""


def test_nearest_parent_label_missing_key_uses_empty_string():
    """Events without key fall back to '' and do not match the prefix."""

    class NoKeyParent:
        __slots__ = ("cpu_parent",)

        def __init__(self, cpu_parent=None):
            self.cpu_parent = cpu_parent

    root = _FakeProfilerEvent(key="ait_nn_module:ok", cpu_parent=None)
    mid = NoKeyParent(cpu_parent=root)
    event = _FakeProfilerEvent(cpu_parent=mid)
    assert kernel_profiler_module._nearest_parent_label(event, "ait_nn_module:") == "ok"


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
    profiler = kernel_profiler_module.ModuleFunctionKernelProfiler()
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
    assert df["time_spent_pct"].tolist() == [600.0 / 1000.0 * 100.0, 300.0 / 1000.0 * 100.0]


def test_describe_results_uses_nan_sample_metrics_for_functions_without_collected_data():
    profiler = kernel_profiler_module.ModuleFunctionKernelProfiler()
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
    profiler = kernel_profiler_module.ModuleFunctionKernelProfiler()
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
    profiler = kernel_profiler_module.ModuleFunctionKernelProfiler()
    module = nn.Identity().to("cuda")
    data = [((torch.randn(1, device="cuda"),), {})]

    profiling_df, function_data = profiler.profile(module, data, warmup_iterations=0)
    summary_df = profiler.describe_results(profiling_df, function_data)

    assert profiling_df.empty
    assert tuple(profiling_df.columns) == tuple(kernel_profiler_module._ProfilingItem.__annotations__)
    assert summary_df.empty
    assert tuple(summary_df.columns) == tuple(kernel_profiler_module._SummaryItem.__annotations__)


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
    profiler = kernel_profiler_module.ModuleFunctionKernelProfiler(function_names=function_names)
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
@pytest.mark.parametrize("teardown_cupti", [None, "0"], ids=["owned", "caller-disabled"])
@pytest.mark.parametrize("warmup_iterations", [0, 3])
def test_profile_repeated_sessions_with_cuda_graphs(monkeypatch, teardown_cupti, warmup_iterations):
    if teardown_cupti is not None:
        monkeypatch.setenv("TEARDOWN_CUPTI", teardown_cupti)
    profiler = ModuleFunctionKernelProfiler()
    net = _MultipleLinearLayerModel().cuda().eval()
    data = net.samples(batch_sizes=[2], device="cuda")
    args, kwargs = data[0]
    with torch.no_grad():
        expected = net(*args, **kwargs)

    # Keep native and Python profiler state across calls within this test.
    for _ in range(10):
        profiling_df, function_data = profiler.profile(net, data, warmup_iterations=warmup_iterations)
        _assert_linear_kernel_attribution(profiling_df)
        assert sum(count for count, _ in function_data["linear"]) == 6

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                net(*args, **kwargs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.no_grad():
            actual = net(*args, **kwargs)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected)


@requires_cuda
def test_profile_module_function_kernels_multiple_samples():
    batch_size, dim = 2, 5
    # given
    profiler = kernel_profiler_module.ModuleFunctionKernelProfiler()
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
    profiler = kernel_profiler_module.ModuleFunctionKernelProfiler()
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
    profiler = kernel_profiler_module.ModuleFunctionKernelProfiler()
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

    assert kernel_profiler_module.get_tensor_size(args, kwargs) == 4 + 4 * 2  # 1x1x4 + 2x2x2 bytes


@requires_cuda
def test_nested_function_calls():
    profiler = kernel_profiler_module.ModuleFunctionKernelProfiler()
    net = _NestedFunctionCallsModel().to("cuda")
    data = net.samples()
    # when
    profiling_df, function_data = profiler.profile(net, data)
    df = profiler.describe_results(profiling_df, function_data)
    observed_function_names = set(df["function_name"].unique().tolist())
    assert observed_function_names == {"multi_head_attention_forward", "linear", "scaled_dot_product_attention"}
