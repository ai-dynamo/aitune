# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module kernel optimization."""

import logging
from collections.abc import Callable
from concurrent.futures import Future
from functools import partial

import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

import aitune.torch.backend.kernels.kernel_optimizer as kernel_optimizer_module
from aitune.torch.backend.kernels.kernel_optimization_plan import KernelOptimizationPlan
from aitune.torch.backend.kernels.kernel_optimizer import KernelOptimizer
from aitune.torch.backend.kernels.kernel_provider import (
    KernelGenerationResult,
    KernelGenerator,
    KernelProvider,
    KernelProviderState,
)
from aitune.torch.backend.kernels.kernel_utils import KernelUtils
from aitune.torch.backend.kernels.module_function_kernel_profiler import ModuleFunctionKernelProfiler
from aitune.torch.module.sample_store import Sample
from aitune.utils.env_vars import AITUNE_KERNEL_GENERATION_TIMEOUT
from tests.utilities.helpers import requires_cuda


class MockKernelProvider(KernelProvider):
    """Kernel provider wrapping a test callable."""

    def __init__(
        self,
        function: Callable,
        events: list[str] | None = None,
        function_name: str = "linear",
    ):
        super().__init__()
        self.function = function
        self.events = events
        self.function_name = function_name
        self.prepare_calls: list[list[Sample]] = []
        self.prepare_grad_enabled: list[bool] = []

    @property
    def supported_function(self) -> str:
        return self.function_name

    @property
    def name(self) -> str:
        return "Mock kernel provider"

    def _prepare(self, samples: list[Sample]) -> bool:
        self.prepare_grad_enabled.append(torch.is_grad_enabled())
        if self.events is not None:
            self.events.append("provider.prepare")
        self.prepare_calls.append(samples)
        return True

    def _infer(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def _to_dict(self) -> dict:
        raise AssertionError("MockKernelProvider is not serializable")

    @classmethod
    def _from_dict(cls, state_dict: dict) -> "MockKernelProvider":
        raise AssertionError("MockKernelProvider is not restorable")


class RecordingFuture(Future[KernelGenerationResult]):
    """Future recording when the optimizer retrieves its result."""

    def __init__(self, events: list[str]):
        super().__init__()
        self.events = events

    def result(self, timeout=None):
        self.events.append("future.result")
        return super().result(timeout)


class MockKernelGenerator(KernelGenerator):
    """Generator returning a predefined completed result or exception."""

    def __init__(
        self,
        provider: Callable | KernelProvider | None,
        *,
        events: list[str] | None = None,
        error: str | None = None,
        exception: Exception | None = None,
        function_name: str = "linear",
        pending: bool = False,
    ):
        if provider is None or isinstance(provider, KernelProvider):
            self.provider = provider
        else:
            self.provider = MockKernelProvider(provider, function_name=function_name)
            self.provider.state = KernelProviderState.READY
        self.events = events
        self.error = error
        self.exception = exception
        self.function_name = function_name
        self.pending = pending
        self.last_future: Future[KernelGenerationResult] | None = None
        self.prepare_calls: list[tuple[str, list[Sample]]] = []
        self.submit_calls: list[tuple[str, list[Sample]]] = []

    def __repr__(self) -> str:
        return "Mock kernel generator"

    def supports_functions(self) -> list[str]:
        return [self.function_name]

    def prepare(self, function: str, samples: list[Sample]) -> bool:
        if self.events is not None:
            self.events.append("generator.prepare")
        self.prepare_calls.append((function, samples))
        return function == self.function_name

    def submit(self, function: str, samples: list[Sample]) -> Future[KernelGenerationResult]:
        if self.events is not None:
            self.events.append("generator.submit")
            future = RecordingFuture(self.events)
        else:
            future = Future()
        self.last_future = future
        self.submit_calls.append((function, samples))
        if self.pending:
            pass
        elif self.exception is not None:
            future.set_exception(self.exception)
        else:
            future.set_result(
                KernelGenerationResult(
                    function=function,
                    provider=self.provider,
                    description="Mock kernel generator",
                    error=self.error,
                )
            )
        return future


class ReluModule(nn.Module):
    """Toy module calling torch.nn.functional.relu."""

    def forward(self, x):
        """Run relu on input tensor."""
        return F.relu(x)


class _MultipleLinearLayerModel(nn.Module):
    """Test module calling the same functional linear kernel multiple times."""

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


class MockKernelUtils(KernelUtils):
    """Stub kernel utils with deterministic validation and benchmark results."""

    def __init__(
        self,
        baseline_latency: float = 100.0,
        provider_latency: float = 200.0,
        events: list[str] | None = None,
    ):
        self.baseline_latency = baseline_latency
        self.provider_latency = provider_latency
        self.events = events
        self.provider: Callable | None = None
        self.validate_calls: list[tuple[Callable, Callable, list[Sample]]] = []
        self.benchmark_calls: list[tuple[Callable, list[Sample]]] = []

    def validate_function(
        self,
        real_function: Callable,
        new_function: Callable,
        samples: list[Sample],
        atol: float = 1e-4,
        rtol: float = 1e-4,
    ) -> list[str]:
        if self.events is not None:
            self.events.append("candidate.validate")
        self.validate_calls.append((real_function, new_function, samples))
        return []

    def benchmark_function(
        self,
        function: Callable,
        samples: list[Sample],
        return_mode: str = "mean",
        warmup: int = 25,
        repeats: int = 100,
    ) -> float:
        if self.events is not None:
            self.events.append("candidate.benchmark")
        self.benchmark_calls.append((function, samples))
        if self.provider is not None and (
            function is self.provider or getattr(function, "function", None) is self.provider
        ):
            return self.provider_latency
        return self.baseline_latency


class MockKernelProfiler(ModuleFunctionKernelProfiler):
    """Kernel profiler recording inputs and returning predefined results."""

    instances: list["MockKernelProfiler"] = []

    def __init__(self, function_names=None, profiling_df=None, function_data=None):
        super().__init__(function_names=function_names)
        self.profiling_df = profiling_df if profiling_df is not None else pd.DataFrame()
        self.function_data = function_data if function_data is not None else {}
        self.function = None
        self.data = None
        self.module = None
        self.__class__.instances.append(self)

    def profile(self, function, data=None, *, module=None, warmup_iterations=3):
        del warmup_iterations
        self.function = function
        self.data = data
        self.module = module
        return self.profiling_df, self.function_data


def test_summary_logs_all_columns(caplog):
    optimizer = KernelOptimizer()
    summary = pd.DataFrame({"function_name": ["linear"], **{f"column_{index}": [index] for index in range(10)}})

    with pd.option_context("display.max_columns", 2, "display.width", 20):
        optimizer._log_summary(summary, ["linear"], [])

    assert "..." not in caplog.text
    assert all(column in caplog.text for column in summary.columns)
    assert "provider" in caplog.text
    assert "generator" in caplog.text
    assert "✓" in caplog.text
    assert "True" not in caplog.text
    assert "False" not in caplog.text


@pytest.mark.parametrize(
    "parameter",
    ["provider_min_time_share_percent", "generator_min_time_share_percent"],
)
@pytest.mark.parametrize("threshold", [-1.0, 100.1, float("inf"), float("nan")])
def test_optimizer_rejects_invalid_time_share_percent(parameter, threshold):
    with pytest.raises(ValueError, match=f"{parameter} must be between 0 and 100"):
        KernelOptimizer(**{parameter: threshold})


def test_make_plan_returns_when_no_kernel_candidates(caplog):
    model = ReluModule()
    optimizer = KernelOptimizer(
        kernel_generators=[MockKernelGenerator(F.relu, function_name="relu")],
        kernel_profiler_factory=MockKernelProfiler,
    )

    with caplog.at_level(logging.INFO, logger="aitune.torch.backend.kernels.kernel_optimizer"):
        plan = optimizer.make_plan(model, [((), {})], module=model)

    messages = [record.getMessage() for record in caplog.records]
    assert any("No kernel candidates found" in message for message in messages)
    assert plan == KernelOptimizationPlan()


def test_make_plan_returns_when_kernel_sources_support_no_functions(caplog, monkeypatch):
    model = ReluModule()
    generator = MockKernelGenerator(F.relu, function_name="relu")
    monkeypatch.setattr(generator, "supports_functions", lambda: [])
    profiler_count = len(MockKernelProfiler.instances)
    optimizer = KernelOptimizer(kernel_generators=[generator], kernel_profiler_factory=MockKernelProfiler)

    with caplog.at_level(logging.INFO, logger="aitune.torch.backend.kernels.kernel_optimizer"):
        plan = optimizer.make_plan(model)

    assert plan == KernelOptimizationPlan()
    assert len(MockKernelProfiler.instances) == profiler_count
    assert "No supported kernel functions" in caplog.text


def test_optimizer_keeps_providers_without_an_availability_check():
    providers = [MockKernelProvider(F.linear), MockKernelProvider(F.linear)]

    optimizer = KernelOptimizer(kernel_providers=providers)

    assert optimizer.kernel_providers == providers


def test_optimizer_normalizes_single_kernel_sources_to_lists():
    provider = MockKernelProvider(F.linear)
    generator = MockKernelGenerator(F.linear)

    optimizer = KernelOptimizer(kernel_providers=provider, kernel_generators=generator)

    assert optimizer.kernel_providers == [provider]
    assert optimizer.kernel_generators == [generator]


def test_make_plan_returns_when_no_kernel_sources_are_configured(caplog):
    model = ReluModule()

    with caplog.at_level(logging.INFO, logger="aitune.torch.backend.kernels.kernel_optimizer"):
        plan = KernelOptimizer().make_plan(model)

    assert plan == KernelOptimizationPlan()
    assert "No available kernel sources" in caplog.text


def test_make_plan_passes_missing_data_to_profiler():
    model = ReluModule()
    optimizer = KernelOptimizer(
        kernel_generators=[MockKernelGenerator(F.relu, function_name="relu")],
        kernel_profiler_factory=MockKernelProfiler,
    )

    optimizer.make_plan(lambda: None, module=model)

    assert MockKernelProfiler.instances[-1].data is None


def test_make_plan_requires_module_scope_for_non_module_function():
    optimizer = KernelOptimizer(kernel_profiler_factory=MockKernelProfiler)

    with pytest.raises(ValueError, match="module must be provided when function is not an nn.Module"):
        optimizer.make_plan(lambda: None)


def test_make_plan_passes_supported_function_names_to_kernel_profiler_factory():
    model = ReluModule()
    original_linear = F.linear
    linear_provider = MockKernelProvider(original_linear)
    conv2d_provider = MockKernelProvider(original_linear, function_name="conv2d")
    softmax_provider = MockKernelProvider(original_linear, function_name="softmax")

    optimizer = KernelOptimizer(
        kernel_providers=[linear_provider, conv2d_provider, softmax_provider],
        kernel_generators=[MockKernelGenerator(F.relu, function_name="relu")],
        kernel_profiler_factory=MockKernelProfiler,
    )

    optimizer.make_plan(model, [((), {})])

    profiler = MockKernelProfiler.instances[-1]
    assert profiler.function_names == {"conv2d", "linear", "relu", "softmax"}
    assert profiler.module is model
    assert optimizer.kernel_providers == [linear_provider, conv2d_provider, softmax_provider]


@pytest.mark.parametrize(("counts", "reduced_counts"), [((6, 4), (3, 2)), ((3, 2), (3, 2))])
def test_prepare_provider_samples_preserves_distribution(counts, reduced_counts):
    sample_a = ((torch.tensor([1.0]),), {})
    sample_b = ((torch.tensor([2.0]),), {})
    samples = [sample_a, sample_b]

    unique_samples, benchmark_samples = KernelOptimizer._prepare_provider_samples(
        list(zip(counts, samples, strict=True))
    )

    assert unique_samples == samples
    assert benchmark_samples == reduced_counts[0] * [sample_a] + reduced_counts[1] * [sample_b]


def test_prepare_provider_samples_handles_empty_function_data():
    unique_samples, benchmark_samples = KernelOptimizer._prepare_provider_samples([])

    assert unique_samples == []
    assert benchmark_samples == []


def test_make_plan_reduces_provider_candidate_benchmark_distribution_by_gcd():
    sample_a = ((torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])), {})
    sample_b = ((torch.tensor([4.0]), torch.tensor([5.0]), torch.tensor([6.0])), {})

    mock_provider = MockKernelProvider(F.linear)
    mock_kernel_utils = MockKernelUtils(baseline_latency=100.0, provider_latency=20.0)
    mock_kernel_utils.provider = mock_provider
    optimizer = KernelOptimizer(kernel_providers=[mock_provider], kernel_utils=mock_kernel_utils)

    plan = optimizer._make_plan(
        {"linear": [(6, sample_a), (4, sample_b)]},
        ["linear"],
        ["linear"],
        [],
    )

    assert plan.providers == (mock_provider,)
    unique_samples = [sample_a, sample_b]
    reduced_samples = [sample_a, sample_a, sample_a, sample_b, sample_b]
    assert mock_provider.prepare_calls == [unique_samples]
    assert mock_provider.prepare_grad_enabled == [False]
    assert mock_kernel_utils.validate_calls == [(F.linear, mock_provider, unique_samples)]
    assert mock_kernel_utils.benchmark_calls == [
        (mock_provider, reduced_samples),
        (F.linear, reduced_samples),
    ]


def test_all_generators_submit_before_provider_evaluation_and_future_collection():
    events = []
    sample = ((torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])), {})

    provider = MockKernelProvider(F.linear, events=events)
    generators = [
        MockKernelGenerator(F.linear, events=events),
        MockKernelGenerator(F.linear, events=events),
    ]
    kernel_utils = MockKernelUtils(baseline_latency=100.0, provider_latency=20.0, events=events)
    kernel_utils.provider = provider
    optimizer = KernelOptimizer(
        kernel_providers=[provider],
        kernel_generators=generators,
        kernel_utils=kernel_utils,
        generator_min_time_share_percent=0.0,
    )

    optimizer._make_plan(
        {"linear": [(1, sample)]},
        ["linear"],
        ["linear"],
        ["linear"],
    )

    submit_indices = [index for index, event in enumerate(events) if event == "generator.submit"]
    assert len(submit_indices) == 2
    assert max(submit_indices) < events.index("provider.prepare")
    assert events.index("candidate.benchmark") < events.index("future.result")
    benchmarked_functions = [function for function, _samples in kernel_utils.benchmark_calls]
    assert benchmarked_functions.count(provider) == 1
    assert benchmarked_functions.count(F.linear) == 1


def test_source_time_share_thresholds_filter_independently():
    linear_sample = ((torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])), {})
    relu_sample = ((torch.tensor([1.0]),), {})
    profiling_df = pd.DataFrame([
        {"function_name": "linear", "module_name": "layer", "kernel_us": 5.0},
        {"function_name": "relu", "module_name": "layer", "kernel_us": 45.0},
        {"function_name": "softmax", "module_name": "layer", "kernel_us": 50.0},
    ])
    function_data = {"linear": [(1, linear_sample)], "relu": [(1, relu_sample)]}
    linear_generator = MockKernelGenerator(F.linear)
    relu_generator = MockKernelGenerator(F.relu, function_name="relu")
    provider = MockKernelProvider(F.linear)
    optimizer = KernelOptimizer(
        kernel_providers=[provider],
        kernel_generators=[linear_generator, relu_generator],
        provider_min_time_share_percent=5.0,
        generator_min_time_share_percent=10.0,
        kernel_utils=MockKernelUtils(),
        kernel_profiler_factory=partial(
            MockKernelProfiler,
            profiling_df=profiling_df,
            function_data=function_data,
        ),
    )

    optimizer.make_plan(ReluModule())

    assert linear_generator.submit_calls == []
    assert relu_generator.submit_calls == [("relu", [relu_sample])]
    assert provider.prepare_calls == [[linear_sample]]


def test_generator_min_time_share_includes_function_at_threshold():
    sample = ((torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])), {})
    profiling_df = pd.DataFrame([
        {"function_name": "linear", "module_name": "layer", "kernel_us": 5.0},
        {"function_name": "softmax", "module_name": "layer", "kernel_us": 95.0},
    ])
    function_data = {"linear": [(1, sample)]}
    generator = MockKernelGenerator(F.linear)
    optimizer = KernelOptimizer(
        kernel_generators=[generator],
        generator_min_time_share_percent=5.0,
        kernel_utils=MockKernelUtils(),
        kernel_profiler_factory=partial(
            MockKernelProfiler,
            profiling_df=profiling_df,
            function_data=function_data,
        ),
    )

    optimizer.make_plan(ReluModule())

    assert generator.submit_calls == [("linear", [sample])]


def test_top_k_is_selected_before_source_support_filtering():
    sample = ((torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])), {})
    provider = MockKernelProvider(F.linear)
    profiling_df = pd.DataFrame([
        {"function_name": "softmax", "module_name": "layer", "kernel_us": 95.0},
        {"function_name": "linear", "module_name": "layer", "kernel_us": 5.0},
    ])
    function_data = {"linear": [(1, sample)]}
    optimizer = KernelOptimizer(
        kernel_providers=[provider],
        top_k=1,
        kernel_utils=MockKernelUtils(),
        kernel_profiler_factory=partial(
            MockKernelProfiler,
            profiling_df=profiling_df,
            function_data=function_data,
        ),
    )

    optimizer.make_plan(ReluModule())

    assert provider.prepare_calls == []


def test_select_best_candidate_returns_none_without_candidates():
    kernel_utils = MockKernelUtils()
    optimizer = KernelOptimizer(kernel_utils=kernel_utils)
    search = kernel_optimizer_module._FunctionSearch(
        real_function=F.linear,
        unique_samples=[],
        benchmark_samples=[],
    )

    assert optimizer._select_best_candidate("linear", search) is None
    assert kernel_utils.benchmark_calls == []


def test_generated_candidate_uses_common_provider_evaluation():
    sample = ((torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])), {})
    generated_function = partial(F.linear)
    generator = MockKernelGenerator(generated_function)
    kernel_utils = MockKernelUtils(baseline_latency=100.0, provider_latency=20.0)
    kernel_utils.provider = generated_function
    optimizer = KernelOptimizer(
        kernel_providers=[],
        kernel_generators=[generator],
        kernel_utils=kernel_utils,
        generator_min_time_share_percent=0.0,
    )

    plan = optimizer._make_plan(
        {"linear": [(1, sample)]},
        ["linear"],
        [],
        ["linear"],
    )

    provider = plan.providers[0]
    assert isinstance(provider, MockKernelProvider)
    assert provider.function is generated_function
    assert kernel_utils.validate_calls == [(F.linear, provider, [sample])]
    assert kernel_utils.benchmark_calls == [
        (provider, [sample]),
        (F.linear, [sample]),
    ]


def test_failed_generation_future_does_not_block_other_generators(caplog):
    sample = ((torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])), {})
    generated_function = partial(F.linear)
    failing_generator = MockKernelGenerator(None, exception=RuntimeError("remote failure"))
    successful_generator = MockKernelGenerator(generated_function)
    kernel_utils = MockKernelUtils(baseline_latency=100.0, provider_latency=20.0)
    kernel_utils.provider = generated_function
    optimizer = KernelOptimizer(
        kernel_providers=[],
        kernel_generators=[failing_generator, successful_generator],
        kernel_utils=kernel_utils,
        generator_min_time_share_percent=0.0,
    )

    plan = optimizer._make_plan(
        {"linear": [(1, sample)]},
        ["linear"],
        [],
        ["linear"],
    )

    provider = plan.providers[0]
    assert isinstance(provider, MockKernelProvider)
    assert provider.function is generated_function
    assert "remote failure" in caplog.text


def test_generation_timeout_defaults_to_environment_configuration():
    assert KernelOptimizer().generation_timeout == AITUNE_KERNEL_GENERATION_TIMEOUT


def test_generation_timeout_keeps_completed_results_and_cancels_unfinished_futures(monkeypatch, caplog):
    sample = ((torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])), {})
    generated_function = partial(F.linear)
    pending_generator = MockKernelGenerator(None, pending=True)
    successful_generator = MockKernelGenerator(generated_function)
    kernel_utils = MockKernelUtils(baseline_latency=100.0, provider_latency=20.0)
    kernel_utils.provider = generated_function
    generation_timeout = 12.5
    optimizer = KernelOptimizer(
        kernel_generators=[pending_generator, successful_generator],
        kernel_utils=kernel_utils,
        generator_min_time_share_percent=0.0,
        generation_timeout=generation_timeout,
    )

    def completed_before_timeout(futures, timeout):
        assert timeout == generation_timeout
        yield next(future for future in futures if future.done())
        raise TimeoutError

    monkeypatch.setattr(kernel_optimizer_module, "as_completed", completed_before_timeout)

    plan = optimizer._make_plan(
        {"linear": [(1, sample)]},
        ["linear"],
        [],
        ["linear"],
    )

    provider = plan.providers[0]
    assert isinstance(provider, MockKernelProvider)
    assert provider.function is generated_function
    assert pending_generator.last_future is not None
    assert pending_generator.last_future.cancelled()
    assert f"timed out after {generation_timeout} seconds" in caplog.text


@requires_cuda
def test_make_plan_multiple_linear_layer_model():
    mock_provider = MockKernelProvider(F.linear)
    mock_kernel_utils = MockKernelUtils(baseline_latency=100.0, provider_latency=20.0)
    mock_kernel_utils.provider = mock_provider

    model = _MultipleLinearLayerModel().to("cuda")
    data = [((torch.randn(2, 5, device="cuda"),), {})]

    optimizer = KernelOptimizer(
        kernel_providers=[mock_provider],
        top_k=2,
        kernel_utils=mock_kernel_utils,
    )  # type: ignore

    plan = optimizer.make_plan(model, data)

    assert mock_provider.prepare_calls
    assert mock_kernel_utils.validate_calls
    assert mock_kernel_utils.benchmark_calls
    assert len(plan.providers) == 1
