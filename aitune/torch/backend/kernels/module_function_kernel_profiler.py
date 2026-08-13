# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Torch profiler for module function kernels."""

import contextvars
import inspect
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from logging import getLogger
from typing import Any, TypedDict, cast

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import wrapt
from torch.autograd.profiler import FunctionEvent
from torch.profiler import ProfilerActivity, profile, record_function

from aitune.torch.module.exact_sample_metadata import ExactSampleMetadata
from aitune.torch.module.locator import Locator
from aitune.torch.module.recording_module import Sample

logger = getLogger(__name__)


MODULE_PREFIX = "ait_nn_module:"
FUNCTIONAL_PREFIX = "ait_torch.nn.functional:"

_cuda_profiler_primed = False


def prime_cuda_profiler() -> None:
    """Prime CUDA profiling with a throwaway first session."""
    global _cuda_profiler_primed  # noqa: PLW0603

    if _cuda_profiler_primed:
        return

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
        pass

    _cuda_profiler_primed = True


class _ProfilingItem(TypedDict):
    """Profiler item: module_name + function + kernel metadata."""

    module_name: str
    module: nn.Module
    function_name: str
    op: str
    kernel: str
    kernel_us: float


class _SummaryItem(TypedDict):
    """Summary of profiling and collected sample data for one function."""

    function_name: str
    calls: int | float
    num_distinct_samples: int | float
    num_modules: int
    time_spent_us: float
    time_spent_pct: float
    tensor_size_MB: float


FunctionData = list[tuple[int, Sample]]


class ModuleFunctionKernelProfiler:
    """Attribute CUDA kernels to module-scoped ``torch.nn.functional`` calls.

    The profiler temporarily instruments module ``forward`` methods and PyTorch
    functional calls, then combines those regions with PyTorch Profiler events.
    It returns kernel-level timing rows and, optionally, the distinct inputs
    observed for selected functional calls.
    """

    def __init__(self, function_names: set[str] | None = None):
        """Initialize the module function kernel profiler.

        Args:
            function_names: Functional call names for which input samples should be collected. This does not filter
                the calls included in profiling results. ``None`` collects samples for every observed function. An
                empty set keeps profiling enabled but skips sample collection for all functions.
        """
        self.function_names = function_names

    def profile(
        self,
        function: Callable,
        data: list[Sample] | None = None,
        *,
        module: nn.Module | None = None,
        warmup_iterations: int = 3,
    ) -> tuple[pd.DataFrame, dict[str, FunctionData]]:  # noqa: C901
        """Profile the module and function kernel candidates.

        Args:
            function: the function to run inference with
            data: the data to use for profiling, if None the function will be called without arguments
            module: the module whose hierarchy should be profiled. If omitted and ``function`` is an ``nn.Module``,
                ``function`` is used as the module.
            warmup_iterations: the number of complete, unprofiled passes over ``data`` before profiling. If ``data``
                is ``None``, each warmup iteration calls ``function`` once.

        Returns:
            A pair containing the profiling dataframe and per-function sample data. Each dataframe row identifies the
            module, functional call, PyTorch operator, CUDA kernel, and kernel duration. Sample data maps a functional
            call name to distinct ``(call_count, sample)`` pairs.

        Raises:
            ValueError: If ``warmup_iterations`` is negative or the module scope cannot be resolved.
        """
        if module is None and isinstance(function, nn.Module):
            module = cast(nn.Module, function)
        if module is None:
            raise ValueError("module must be provided when function is not an nn.Module.")
        named_modules = list(module.named_modules())

        if warmup_iterations < 0:
            raise ValueError("warmup_iterations must be greater than or equal to 0")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to profile module and function kernel candidates.")

        prime_cuda_profiler()  # required by some torch versions
        if warmup_iterations:
            with torch.no_grad():
                for _ in range(warmup_iterations):
                    self._run_inference(function, data)
            torch.cuda.synchronize()

        self._initialize()

        with self._record_module_forwards(named_modules):
            with self._record_functional_calls():
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    with torch.no_grad():
                        self._run_inference(function, data)

        torch.cuda.empty_cache()

        profiling_data, function_data = self._match_events_with_data(prof.profiler.function_events, named_modules)
        if profiling_data:
            profiling_df = pd.DataFrame(profiling_data)
        else:
            profiling_df = pd.DataFrame(columns=tuple(_ProfilingItem.__annotations__))
        return profiling_df, function_data

    def describe_results(
        self,
        profiling_df: pd.DataFrame,
        function_data: dict[str, FunctionData],
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Describe the profiling and function data.

        Args:
            profiling_df: the profiling data to describe
            function_data: the function data to describe
            top_k: the number of top-k functions to describe

        Returns:
            A dataframe summarizing call counts, distinct samples, modules, kernel time, time share, and input tensor
            size for up to ``top_k`` functional calls. Time share is relative to the functions included in the summary.
        """
        if profiling_df.empty:
            return pd.DataFrame(columns=tuple(_SummaryItem.__annotations__))

        top_k_function_times = profiling_df.groupby("function_name")["kernel_us"].sum().nlargest(top_k)
        total_time_spent_us = top_k_function_times.sum()

        summary: list[_SummaryItem] = []
        for function_name, time_spent_us in top_k_function_times.items():
            df_function = profiling_df.query("function_name == @function_name")
            data = function_data.get(function_name)
            if data is None:
                calls = float("nan")
                num_distinct_samples = float("nan")
                tensor_size_mb = float("nan")
            else:
                calls = sum(count for count, _ in data)
                num_distinct_samples = len(data)
                tensor_size_mb = sum(get_tensor_size(*sample) for _, sample in data) / 1024**2
            summary.append(
                _SummaryItem(
                    function_name=function_name,
                    calls=calls,
                    num_distinct_samples=num_distinct_samples,
                    num_modules=len(df_function["module_name"].unique()),
                    time_spent_us=time_spent_us,
                    time_spent_pct=time_spent_us / total_time_spent_us * 100.0 if total_time_spent_us else 0.0,
                    tensor_size_MB=tensor_size_mb,
                )
            )

        return pd.DataFrame(summary)

    def get_functions_to_patch(self, module=F) -> list[tuple[str, Any]]:
        """Return available functions to patch from a module."""
        result: list[tuple[str, Any]] = []
        for func_name, func in vars(module).items():
            if self._should_patch(module, func_name, func):
                result.append((func_name, func))
        return result

    def _initialize(self):
        """Initialize the module function kernel profiler."""
        self.module_stack = contextvars.ContextVar("current_module_stack", default=())
        self.data_cache: dict[ExactSampleMetadata, Sample] = {}
        self.func_data_distribution: dict[str, dict[ExactSampleMetadata, int]] = defaultdict(lambda: defaultdict(int))

    @contextmanager
    def _record_module_forwards(self, named_modules: Iterable[tuple[str, nn.Module]]) -> Generator[None, None, None]:
        """Context manager to patch the module forwards.

        The module stack is used to track the current module hierarchy.
        The module sentinel is used to label the module forward calls for profiling.

        Args:
            named_modules: The named modules to patch.
        """
        originals = {}
        try:
            for name, module in named_modules:
                originals[name] = module.forward
                module_sentinel = f"{MODULE_PREFIX}{name}"

                def make_wrapper(module_sentinel, name):
                    def wrapper(wrapped, instance, args, kwargs):
                        stack = self.module_stack.get()
                        token = self.module_stack.set(stack + (name,))

                        try:
                            with record_function(module_sentinel):
                                return wrapped(*args, **kwargs)
                        finally:
                            self.module_stack.reset(token)

                    return wrapper

                module.forward = wrapt.FunctionWrapper(module.forward, make_wrapper(module_sentinel, name))

            yield
        finally:
            for name, module in named_modules:
                module.forward = originals[name]

    @contextmanager
    def _record_functional_calls(self, module=F) -> Generator[None, None, None]:
        """Context manager to patch the Torch functional calls.

        Args:
            module: The functional module to patch.
        """
        originals = {}

        try:
            for func_name, func in self.get_functions_to_patch(module):
                originals[func_name] = func
                functional_sentinel = f"{FUNCTIONAL_PREFIX}{func_name}"
                try:
                    function_signature = inspect.signature(func)
                except (TypeError, ValueError):
                    function_signature = None

                def make_wrapper(functional_sentinel, function_name, function_signature):
                    def wrapper(wrapped, instance, args, kwargs):
                        stack = self.module_stack.get()
                        current_module = stack[-1] if stack else None
                        if current_module is not None and self._collects_function_data(function_name):
                            # record only data within observed modules
                            if function_signature is None:
                                inputs = {"args": args, "kwargs": kwargs}
                            else:
                                inputs = function_signature.bind(*args, **kwargs).arguments
                            new_metadata = ExactSampleMetadata.from_inputs(inputs)
                            if new_metadata not in self.data_cache:
                                self.data_cache[new_metadata] = (args, kwargs)
                            self.func_data_distribution[function_name][new_metadata] += 1

                        with record_function(functional_sentinel):
                            return wrapped(*args, **kwargs)

                    return wrapper

                setattr(
                    module,
                    func_name,
                    wrapt.FunctionWrapper(func, make_wrapper(functional_sentinel, func_name, function_signature)),
                )

            yield
        finally:
            for func_name, func in originals.items():
                setattr(module, func_name, func)

    def _match_events_with_data(
        self,
        function_events: list[FunctionEvent],
        named_modules: list[tuple[str, nn.Module]],
    ) -> tuple[list[_ProfilingItem], dict[str, FunctionData]]:
        """Match the profiler function events with data.

        It uses profiler data, specifically kernels attribute TO filter function events. Next it walks through
        `cpu_parent` property to find the nearest parent label that starts with the module or function prefix.
        """
        profiling_data: list[_ProfilingItem] = []
        function_data: dict[str, FunctionData] = {}
        matched_func_names: set[str] = set()

        modules_dict = dict(named_modules)

        for e in function_events:
            kernels = getattr(e, "kernels", None) or []
            if not kernels:
                continue

            module_name = _nearest_parent_label(e, MODULE_PREFIX)
            func_name = _nearest_parent_label(e, FUNCTIONAL_PREFIX)

            if module_name is None or func_name is None:
                # skip empty module/function, kernels like Memcpy
                continue

            if "::" not in e.key:
                # skip non-module/function kernels
                continue

            module = modules_dict[module_name]
            if self._collects_function_data(func_name):
                matched_func_names.add(func_name)

            for k in kernels:
                profiling_data.append(
                    _ProfilingItem(
                        module_name=module_name,
                        module=module,
                        function_name=func_name,
                        op=e.key,
                        kernel=k.name,
                        kernel_us=k.duration,
                    )
                )

        for func_name in matched_func_names:
            function_data[func_name] = self._rewrite_func_data(func_name)

        return profiling_data, function_data

    def _rewrite_func_data(self, func_name: str) -> FunctionData:
        """Rewrite func data from dict[ExactSampleMetadata, int] to list[tuple[int, Sample]]."""
        return [(count, self.data_cache[meta]) for meta, count in self.func_data_distribution[func_name].items()]

    def _collects_function_data(self, func_name: str) -> bool:
        """Return whether sample data should be collected for the function."""
        return self.function_names is None or func_name in self.function_names

    def _should_patch(self, module, func_name: str, func: Any) -> bool:
        """Check if a function should be patched."""
        if func_name.startswith("_"):
            return False
        if not (inspect.isfunction(func) or inspect.isbuiltin(func)):
            return False
        return getattr(func, "__module__", None) in {module.__name__, "torch", "torch._C._nn"}

    @staticmethod
    def _run_inference(function: Callable, data: list[Sample] | None) -> None:
        """Run one complete inference iteration."""
        if data is None:
            function()
            return
        for args, kwargs in data:
            function(*args, **kwargs)


def get_tensor_size(args, kwargs):
    """Get the total size of tensors in args and kwargs."""
    tensor_size = 0
    for _, tensor in Locator.find_leaves(args, only_tensors=True):
        if torch.is_tensor(tensor):
            tensor_size += tensor.numel() * tensor.element_size()
    for _, tensor in Locator.find_leaves(kwargs, only_tensors=True):
        if torch.is_tensor(tensor):
            tensor_size += tensor.numel() * tensor.element_size()
    return tensor_size


def _nearest_parent_label(event, prefix):
    """Find the nearest parent label that starts with the prefix.

    It uses labels marked with record_function context manager.
    """
    p = getattr(event, "cpu_parent", None)
    while p is not None:
        key = getattr(p, "key", "")
        if key.startswith(prefix):
            return key.removeprefix(prefix)
        p = getattr(p, "cpu_parent", None)

    return None
