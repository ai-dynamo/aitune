# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel optimizer based on module function kernel profiling."""

from collections.abc import Callable
from concurrent.futures import Future, as_completed
from dataclasses import dataclass, field
from itertools import chain
from logging import getLogger
from math import gcd
from typing import TypeVar

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from aitune.torch.backend.kernels.kernel_optimization_plan import KernelOptimizationPlan
from aitune.torch.backend.kernels.kernel_provider import KernelGenerationResult, KernelGenerator, KernelProvider
from aitune.torch.backend.kernels.kernel_utils import KernelUtils
from aitune.torch.backend.kernels.module_function_kernel_profiler import (
    FunctionData,
    ModuleFunctionKernelProfiler,
)
from aitune.torch.module.sample_store import Sample
from aitune.utils.env_vars import AITUNE_KERNEL_GENERATION_TIMEOUT
from aitune.utils.validation import in_range

logger = getLogger(__name__)

KernelSource = KernelProvider | KernelGenerator
KernelSourceType = TypeVar("KernelSourceType", KernelProvider, KernelGenerator)


@dataclass
class _KernelCandidate:
    """A kernel candidate."""

    provider: KernelProvider
    """The selected runtime kernel provider."""
    latency: float
    """The latency of the candidate."""
    description: str
    """The description of the candidate."""


@dataclass
class _FunctionSearch:
    """Samples and accumulated candidates for one PyTorch function."""

    real_function: Callable
    unique_samples: list[Sample]
    benchmark_samples: list[Sample]
    candidates: list[_KernelCandidate] = field(default_factory=list)


@dataclass(frozen=True)
class _GenerationTask:
    """One submitted generator task and its expected function."""

    function_name: str
    generator_name: str
    future: Future[KernelGenerationResult]


class KernelOptimizer:
    """Profiles and replaces top-k ``torch.nn.functional`` kernels in a module.

    ``make_plan(function, data, module=...)`` profiles forwards, starts asynchronous
    generators, benchmarks static providers while generation runs, collects and
    benchmarks generated candidates, and returns the selected runtime plan.
    """

    def __init__(
        self,
        kernel_providers: KernelProvider | list[KernelProvider] | None = None,
        kernel_generators: KernelGenerator | list[KernelGenerator] | None = None,
        top_k: int = 5,
        kernel_profiler_factory: Callable[..., ModuleFunctionKernelProfiler] = ModuleFunctionKernelProfiler,
        kernel_utils: KernelUtils | None = None,
        provider_min_time_share_percent: float = 0.0,
        generator_min_time_share_percent: float = 10.0,
        generation_timeout: float = AITUNE_KERNEL_GENERATION_TIMEOUT,
    ):
        """Initialize the kernel optimizer.

        Args:
            kernel_providers: static kernel providers. A single provider is normalized to a list.
            kernel_generators: asynchronous kernel generators. A single generator
                is normalized to a list. Generators are submitted before any
                static provider is evaluated.
            top_k: the number of top kernels to optimize, default is 5
            kernel_profiler_factory: factory used to create the module function kernel profiler. The default is
                ``ModuleFunctionKernelProfiler``. It receives ``function_names`` supported by kernel providers or
                kernel generators.
            kernel_utils: the kernel utilities to use, default is KernelUtils
            provider_min_time_share_percent: minimum percentage of total profiled kernel time that a function must
                account for before static providers are evaluated for it. Must be between 0 and 100. The default of
                0 preserves evaluation for every supported function in the top-k.
            generator_min_time_share_percent: minimum percentage of total profiled kernel time that a function must
                account for before generators are submitted for it. Must be between 0 and 100. The default of 10
                limits generation to functions with a significant profiling share.
            generation_timeout: maximum time in seconds to wait for submitted kernel generators. Defaults to
                ``AITUNE_KERNEL_GENERATION_TIMEOUT``, or six hours when the environment variable is unset. Unfinished
                generator futures are cancelled or ignored after the timeout.
        """
        if kernel_providers is None:
            kernel_providers = []
        elif isinstance(kernel_providers, KernelProvider):
            kernel_providers = [kernel_providers]
        if kernel_generators is None:
            kernel_generators = []
        elif isinstance(kernel_generators, KernelGenerator):
            kernel_generators = [kernel_generators]

        in_range(provider_min_time_share_percent, min_value=0, max_value=100, name="provider_min_time_share_percent")
        in_range(generator_min_time_share_percent, min_value=0, max_value=100, name="generator_min_time_share_percent")

        self.kernel_providers = kernel_providers
        self.kernel_generators = kernel_generators
        self.top_k = top_k
        self.provider_min_time_share_percent = provider_min_time_share_percent
        self.generator_min_time_share_percent = generator_min_time_share_percent
        self.generation_timeout = generation_timeout
        self.kernel_profiler_factory = kernel_profiler_factory
        self.kernel_utils = kernel_utils if kernel_utils is not None else KernelUtils()

    def make_plan(
        self,
        function: Callable,
        data: list[Sample] | None = None,
        *,
        module: nn.Module | None = None,
    ) -> KernelOptimizationPlan:
        """Optimize a module by replacing the top-k kernels with the best ones.

        Args:
            function: the function to run inference with
            data: the data to use for optimization, if None the function will be called without arguments
            module: the module whose hierarchy should be optimized. If omitted and ``function`` is an ``nn.Module``,
                ``function`` is used as the module.

        Returns:
            A plan containing the selected runtime kernel providers.

        Raises:
            ValueError: If the module scope cannot be resolved.
        """
        if module is None and isinstance(function, nn.Module):
            module = function
        if module is None:
            raise ValueError("module must be provided when function is not an nn.Module.")

        if not self.kernel_providers and not self.kernel_generators:
            logger.info("No available kernel sources.")
            return KernelOptimizationPlan()

        supported_function_names = self._supported_function_names()
        if not supported_function_names:
            logger.info("No supported kernel functions.")
            return KernelOptimizationPlan()
        kernel_profiler = self.kernel_profiler_factory(function_names=supported_function_names)

        profiling_df, function_data = kernel_profiler.profile(function, data, module=module)
        if profiling_df.empty:
            logger.info("No kernel candidates found.")
            return KernelOptimizationPlan()
        summary_df = kernel_profiler.describe_results(profiling_df, function_data, self.top_k)

        top_k_functions = summary_df["function_name"].tolist()
        provider_function_names = self._select_source_functions(
            summary_df,
            self.kernel_providers,
            self.provider_min_time_share_percent,
        )
        generator_function_names = self._select_source_functions(
            summary_df,
            self.kernel_generators,
            self.generator_min_time_share_percent,
        )
        self._log_summary(summary_df, provider_function_names, generator_function_names)

        return self._make_plan(function_data, top_k_functions, provider_function_names, generator_function_names)

    def _supported_function_names(self) -> set[str]:
        """Return function names supported by providers or generators."""
        function_names = set()
        for source in self._kernel_sources():
            for function_name in self._supported_functions(source):
                function_names.add(function_name)
        return function_names

    def _select_source_functions(
        self,
        summary_df: pd.DataFrame,
        sources: list[KernelSourceType],
        min_time_share_percent: float,
    ) -> list[str]:
        """Select top-k functions supported by a source and meeting its time-share threshold."""
        supported_functions = {function for source in sources for function in self._supported_functions(source)}
        selected = summary_df["function_name"].isin(supported_functions) & summary_df["time_spent_pct"].ge(
            min_time_share_percent
        )
        return summary_df.loc[selected, "function_name"].tolist()

    def _log_summary(
        self,
        summary_df: pd.DataFrame,
        provider_function_names: list[str],
        generator_function_names: list[str],
    ) -> None:
        """Log profiler results with markers for the selected kernel sources."""
        selected = {True: "✓", False: ""}
        summary_df["provider"] = summary_df["function_name"].isin(provider_function_names).map(selected)
        summary_df["generator"] = summary_df["function_name"].isin(generator_function_names).map(selected)
        with pd.option_context("display.max_columns", None, "display.width", None):
            logger.info("Summary of the kernel profiler:\n%s", summary_df.round(2))

    def _log_kernel_candidates_results(
        self,
        func_name: str,
        candidates: list[_KernelCandidate],
        baseline_latency: float,
    ):
        """Log candidate latencies and their improvement over the baseline."""
        df = pd.DataFrame([(c.description, c.latency) for c in candidates], columns=["Candidate", "Latency [ms]"])
        df["Improvement"] = baseline_latency / df["Latency [ms]"]
        with pd.option_context("display.max_columns", None, "display.width", None):
            logger.info("Kernel candidates results for %s:\n%s", func_name, df.set_index("Candidate").round(2))
        logger.info("Baseline latency: %.2f [ms]", baseline_latency)

    def _make_plan(
        self,
        function_data: dict[str, FunctionData],
        top_k_functions: list[str],
        provider_functions: list[str],
        generator_functions: list[str],
    ) -> KernelOptimizationPlan:
        """Evaluate kernel candidates and return the best providers.

        All generator tasks are scheduled first. While asynchronous generation is running, candidates from static
        providers are prepared and benchmarked. Generated results are then collected and evaluated using the same
        validation and benchmarking path. Finally, the fastest candidate for each function is selected when it
        outperforms the original PyTorch function.

        Args:
            function_data: Collected input samples for each supported function.
            top_k_functions: Globally selected top-k functions in profiling order.
            provider_functions: Top-k functions selected for static providers.
            generator_functions: Top-k functions selected for asynchronous generators.

        Returns:
            A runtime plan containing the fastest providers that outperform their baselines.
        """
        selected_functions = set(provider_functions) | set(generator_functions)
        searches = self._prepare_searches(
            function_data,
            [function for function in top_k_functions if function in selected_functions],
        )
        generation_tasks = self._submit_generation_tasks(searches, generator_functions)
        self._evaluate_provider_candidates(searches, provider_functions)
        self._evaluate_generation_results(searches, generation_tasks)

        providers = []
        for func_name, search in searches.items():
            best_candidate = self._select_best_candidate(func_name, search)
            if best_candidate is not None:
                providers.append(best_candidate.provider)

        return KernelOptimizationPlan(tuple(providers))

    def _prepare_searches(
        self,
        function_data: dict[str, FunctionData],
        top_k_functions: list[str],
    ) -> dict[str, _FunctionSearch]:
        """Prepare each selected function's samples exactly once."""
        searches = {}
        for func_name in top_k_functions:
            unique_samples, benchmark_samples = self._prepare_provider_samples(function_data[func_name])
            searches[func_name] = _FunctionSearch(
                real_function=getattr(F, func_name),
                unique_samples=unique_samples,
                benchmark_samples=benchmark_samples,
            )
        return searches

    def _submit_generation_tasks(
        self,
        searches: dict[str, _FunctionSearch],
        generator_functions: list[str],
    ) -> list[_GenerationTask]:
        """Submit all supported generator-function pairs before static work."""
        tasks = []
        for func_name in generator_functions:
            search = searches[func_name]
            for generator in self._generators_for_function(func_name):
                generator_name = str(generator)
                try:
                    if generator.prepare(func_name, search.unique_samples):
                        future = generator.submit(func_name, search.unique_samples)
                        tasks.append(_GenerationTask(func_name, generator_name, future))
                    else:
                        logger.info("Generator validation failed for %s", generator_name)
                except Exception as error:  # noqa: BLE001 - isolate one external generator
                    logger.warning("Generator submission failed for %s: %s", generator_name, error)
        return tasks

    def _evaluate_generation_results(
        self,
        searches: dict[str, _FunctionSearch],
        tasks: list[_GenerationTask],
    ) -> None:
        """Evaluate generated candidates sequentially as their futures finish."""
        tasks_by_future = {task.future: task for task in tasks}
        try:
            for future in as_completed(tasks_by_future, timeout=self.generation_timeout):
                task = tasks_by_future[future]
                try:
                    result = future.result()
                    self._evaluate_generation_result(searches, task, result)
                except Exception as error:  # noqa: BLE001 - isolate one asynchronous task
                    logger.warning("Kernel generation failed in %s: %s", task.generator_name, error)
        except TimeoutError:
            unfinished_tasks = [task for future, task in tasks_by_future.items() if not future.done()]
            for task in unfinished_tasks:
                task.future.cancel()
            logger.warning(
                "Kernel generation timed out after %s seconds; ignoring %d unfinished task(s): %s",
                self.generation_timeout,
                len(unfinished_tasks),
                ", ".join(task.generator_name for task in unfinished_tasks),
            )

    def _evaluate_candidate(
        self,
        search: _FunctionSearch,
        provider: KernelProvider,
        description: str,
    ) -> None:
        """Run the common correctness validation and benchmark for one candidate."""
        try:
            errors = self.kernel_utils.validate_function(
                search.real_function,
                provider,
                search.unique_samples,
            )
            if errors:
                logger.info("Validation failed for %s, errors:\n%s", description, "\n".join(errors))
            else:
                latency = self.kernel_utils.benchmark_function(provider, search.benchmark_samples)
                search.candidates.append(_KernelCandidate(provider, latency, description))
        except Exception as error:  # noqa: BLE001 - isolate one candidate
            logger.warning("Candidate evaluation failed for %s: %s", description, error)

    def _evaluate_provider_candidates(
        self,
        searches: dict[str, _FunctionSearch],
        provider_functions: list[str],
    ) -> None:
        """Prepare and benchmark all provider candidates while generators run."""
        for func_name in provider_functions:
            search = searches[func_name]
            logger.info("Searching for static kernels for %s", func_name)
            for provider in self._providers_for_function(func_name):
                try:
                    description = str(provider)
                    with torch.no_grad():
                        if provider.prepare(search.unique_samples):
                            self._evaluate_candidate(search, provider, description)
                        else:
                            logger.info("Preparation failed for %s", description)
                except Exception as error:  # noqa: BLE001 - isolate one external provider
                    logger.warning("Static kernel source failed for %s: %s", provider, error)

    def _evaluate_generation_result(
        self,
        searches: dict[str, _FunctionSearch],
        task: _GenerationTask,
        result: KernelGenerationResult,
    ) -> None:
        """Evaluate one completed generation result."""
        if result.succeeded and result.provider is not None:
            self._evaluate_candidate(
                searches[task.function_name],
                result.provider,
                result.description,
            )
        else:
            logger.warning(
                "Generator %s produced no candidate for %s: %s",
                task.generator_name,
                task.function_name,
                result.error or "no provider",
            )

    def _select_best_candidate(self, func_name: str, search: _FunctionSearch) -> _KernelCandidate | None:
        """Select the fastest valid candidate when it beats the single baseline measurement."""
        if not search.candidates:
            return None
        try:
            baseline_latency = self.kernel_utils.benchmark_function(search.real_function, search.benchmark_samples)
            self._log_kernel_candidates_results(func_name, search.candidates, baseline_latency)
            best_candidate = min(search.candidates, key=lambda candidate: candidate.latency)
            if best_candidate.latency < baseline_latency:
                logger.info("Best candidate is faster than baseline: %s", best_candidate.description)
                return best_candidate
        except Exception as error:  # noqa: BLE001 - baseline failure affects only this function
            logger.warning("Baseline benchmark failed for %s: %s", func_name, error)
        return None

    def _providers_for_function(self, func_name: str) -> list[KernelProvider]:
        """Return static providers supporting one function."""
        return self._kernel_sources_for_function(self.kernel_providers, func_name)

    def _generators_for_function(self, func_name: str) -> list[KernelGenerator]:
        """Return dynamic generators supporting one function."""
        return self._kernel_sources_for_function(self.kernel_generators, func_name)

    def _kernel_sources(self) -> list[KernelSource]:
        """Return all available static and dynamic kernel sources."""
        return [*self.kernel_providers, *self.kernel_generators]

    @staticmethod
    def _kernel_sources_for_function(
        sources: list[KernelSourceType],
        func_name: str,
    ) -> list[KernelSourceType]:
        """Return sources supporting one function."""
        return [source for source in sources if func_name in KernelOptimizer._supported_functions(source)]

    @staticmethod
    def _supported_functions(source: KernelSource) -> list[str]:
        """Return function names supported by a provider or generator."""
        if isinstance(source, KernelProvider):
            return [source.supported_function]
        return source.supports_functions()

    @staticmethod
    def _prepare_provider_samples(function_data: FunctionData) -> tuple[list[Sample], list[Sample]]:
        """Return unique validation samples and a GCD-reduced benchmark distribution.

        By dividing counters by the GCD of the counters, we can reduce the number of samples
        to a minimum while maintaining the same distribution.

        Args:
            function_data: the function data to prepare the provider samples for

        Returns:
            a tuple of unique validation samples and a GCD-reduced benchmark distribution
        """
        if not function_data:
            return [], []

        unique_samples = [sample for _, sample in function_data]
        distribution_gcd = 0
        for counter, _ in function_data:
            distribution_gcd = gcd(distribution_gcd, counter)

        benchmark_samples = list(
            chain.from_iterable((counter // distribution_gcd) * [sample] for counter, sample in function_data)
        )
        return unique_samples, benchmark_samples
