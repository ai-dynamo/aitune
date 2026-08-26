# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Latency budget tune strategy."""

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any

from aitune.torch.backend import Backend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import SampleStore
from aitune.torch.task.profiling import ProfilingConfig
from aitune.torch.task.profiling.events import ProfilingResultEvent, get_inference_events
from aitune.torch.task.profiling.measuring_stop_strategy import MeasuringStopStrategy
from aitune.torch.task.profiling.metrics import get_latency, get_throughput
from aitune.torch.task.profiling.profiling import ProfilingStatus, profile_backend
from aitune.torch.task.profiling.profiling_stop_strategy import ProfilingStopStrategy
from aitune.torch.tune_strategy.profiling_tune_strategy import (
    BackendProfilingResult,
    ProfilingTuneStrategy,
    _TuneCandidate,
)
from aitune.utils import validation


@dataclass(kw_only=True)
class LatencyBudgetProfilingResult(BackendProfilingResult):
    """Profiling result for latency-budget selection."""

    selected_batch_size: int
    throughput: float
    latency: float

    @property
    def metric(self) -> float:
        """Returns throughput as the comparison metric."""
        return self.throughput

    def to_json_dict(self, metric_label: str) -> dict[str, int | float]:
        """Returns strategy result fields for latency-budget profiling."""
        result = super().to_json_dict(metric_label)
        result["latency"] = self.latency
        return result


@dataclass
class _LatencyBudgetProfilingStopStrategy(ProfilingStopStrategy):
    latency_budget_ms: float
    measurement_stop_strategy: MeasuringStopStrategy
    base_strategy: ProfilingStopStrategy

    def should_stop(self, results: list[ProfilingResultEvent]) -> bool:
        if self.base_strategy.should_stop(results):
            return True

        measured_events = self.measurement_stop_strategy.get_events(results)
        return bool(measured_events) and get_latency(measured_events) > self.latency_budget_ms


class LatencyBudgetStrategy(ProfilingTuneStrategy):
    """Searches for max throughput while satisfying a latency budget.

    Each backend is profiled across the configured batch-size sweep. For each backend,
    the strategy keeps only batch sizes whose mean latency is less than or equal to
    ``latency_budget_ms``, then uses the highest-throughput remaining batch size as
    that backend's score. The selected backend is the one with maximum throughput
    among all latency-compliant candidates.

    TorchEager is profiled as a performance-validation baseline when validation is enabled.
    If no user-provided backend satisfies the latency budget, the strategy raises.
    """

    _title = "Latency Budget Strategy"
    _description = "evaluate all backends, return max throughput backend within latency budget"
    _metric_label = "throughput"
    _metric_unit = "samples/s"
    _value_fmt = ".2f"

    def __init__(self, latency_budget_ms: float, *args, **kwargs):
        """Initializes LatencyBudgetStrategy.

        Args:
            latency_budget_ms: Maximum allowed mean latency in milliseconds.
            args: Positional arguments passed to ``ProfilingTuneStrategy``.
            kwargs: Keyword arguments passed to ``ProfilingTuneStrategy``.
        """
        validation.positive(latency_budget_ms)
        super().__init__(*args, **kwargs)
        self.latency_budget_ms = latency_budget_ms

    def _measure(
        self,
        backend: Backend,
        name: str,
        graph_spec: GraphSpec,
        samples: SampleStore,
        profiling_cfg: ProfilingConfig,
    ) -> LatencyBudgetProfilingResult:
        """Profiles the backend and returns the max-throughput result within the latency budget."""
        profiling_results = profile_backend(backend, name, graph_spec, samples, profiling_cfg)
        if profiling_results.status != ProfilingStatus.Status.SUCCESS:
            raise profiling_results.error or RuntimeError("Profiling failed")

        candidate = _find_best_latency_budget_candidate(
            get_inference_events(profiling_results.results.entries),
            profiling_cfg.measurement_stop_strategy,
            self.latency_budget_ms,
        )
        if candidate is None:
            raise RuntimeError(
                f"No profile result satisfied latency budget {self.latency_budget_ms:.3f} ms "
                f"for backend {backend.describe()}"
            )

        return candidate

    def _is_better(self, result: BackendProfilingResult, other: BackendProfilingResult) -> bool:
        return result.metric > other.metric

    def _speedup(self, result: BackendProfilingResult, baseline_result: BackendProfilingResult) -> float:
        return result.metric / baseline_result.metric

    def _get_profiling_config(self, batching: bool, max_batch_size: int) -> ProfilingConfig:
        profiling_config = super()._get_profiling_config(batching, max_batch_size)
        return replace(
            profiling_config,
            profiling_stop_strategy=_LatencyBudgetProfilingStopStrategy(
                latency_budget_ms=self.latency_budget_ms,
                measurement_stop_strategy=profiling_config.measurement_stop_strategy,
                base_strategy=profiling_config.profiling_stop_strategy,
            ),
        )

    def _resolve_winner(self, best: _TuneCandidate | None) -> _TuneCandidate:
        if best is None:
            raise RuntimeError(f"No backend satisfied latency budget {self.latency_budget_ms:.3f} ms")
        return super()._resolve_winner(best)

    def _describe_parts(self) -> list[str]:
        return [
            *super()._describe_parts(),
            f"latency_budget_ms: {self.latency_budget_ms:g}",
        ]

    def to_json_dict(self) -> dict[str, Any]:
        """Returns config dict for the strategy."""
        result = super().to_json_dict()
        result["latency_budget_ms"] = self.latency_budget_ms
        return result


def _find_best_latency_budget_candidate(
    profiling_results: list[ProfilingResultEvent],
    measuring_stop_strategy: MeasuringStopStrategy,
    latency_budget_ms: float,
) -> LatencyBudgetProfilingResult | None:
    """Find the highest-throughput profiled batch size that satisfies the latency budget."""
    events_by_batch_size = defaultdict(list)
    for event in profiling_results:
        events_by_batch_size[event.batch_size].append(event)

    best: LatencyBudgetProfilingResult | None = None
    for batch_size, events in events_by_batch_size.items():
        measured_events = measuring_stop_strategy.get_events(events)
        if not measured_events:
            continue
        latency = get_latency(measured_events)
        if latency <= latency_budget_ms:
            candidate = LatencyBudgetProfilingResult(
                selected_batch_size=batch_size,
                throughput=get_throughput(measured_events, batch_size),
                latency=latency,
            )
            if best is None or candidate.throughput > best.throughput:
                best = candidate

    return best
