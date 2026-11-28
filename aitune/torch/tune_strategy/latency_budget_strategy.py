# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Latency budget tune strategy."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any

from aitune.torch.backend import Backend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import SampleStore
from aitune.torch.task.profiling import ProfilingConfig
from aitune.torch.task.profiling.events import ProfilingResultEvent
from aitune.torch.task.profiling.measuring_stop_strategy import MeasuringStopStrategy
from aitune.torch.task.profiling.metrics import BoundaryLatencies, get_latency, throughput_speedup
from aitune.torch.task.profiling.profiling_stop_strategy import ProfilingStopStrategy
from aitune.torch.tune_strategy.profiling_tune_strategy import (
    BackendProfilingResult,
    ProfilingTuneStrategy,
    _TuneCandidate,
)
from aitune.torch.utils.shape_samples import shape_boundary_samples
from aitune.utils import validation

if TYPE_CHECKING:
    from aitune.torch.task.profiling.profiling import ProfilingResults, ProfilingStatus
from aitune.torch.tune_strategy.profiling_tune_strategy import (
    BackendProfilingResult,
    ProfilingTuneStrategy,
    _TuneCandidate,
)
from aitune.torch.utils.shape_samples import shape_boundary_samples
from aitune.utils import validation


@dataclass(kw_only=True)
class LatencyBudgetProfilingResult(BackendProfilingResult):
    """Profiling result for latency-budget selection."""

    selected_batch_size: int
    throughput: float
    latency: float

    @property
    def metric(self) -> float:
        """Return the min-shape throughput used in existing reports and speedup logs."""
        return self.throughput

    @property
    def selection_score(self) -> float:
        """Rank budget-compliant batches by both shapes without changing reported values."""
        if self.boundary_latencies is None:
            return self.metric
        return self.boundary_latencies.throughput_score(self.selected_batch_size)

    def to_json_dict(self, metric_label: str) -> dict[str, int | float]:
        """Returns strategy result fields for latency-budget profiling."""
        result = super().to_json_dict(metric_label)
        result["latency"] = self.latency
        return result


@dataclass
class _LatencyBudgetProfilingStopStrategy(ProfilingStopStrategy):
    """Stop on any sample's budget violation, otherwise consult primary-sample saturation."""

    latency_budget_ms: float
    measurement_stop_strategy: MeasuringStopStrategy
    base_strategy: ProfilingStopStrategy

    def should_stop_samples(self, samples: Sequence[list[ProfilingResultEvent]]) -> bool:
        """Require every sample to meet the budget; saturation uses the primary one."""
        for events in samples:
            measured = self.measurement_stop_strategy.get_events(events)
            if measured and get_latency(measured) > self.latency_budget_ms:
                return True
        return self.base_strategy.should_stop(samples[0])

    def should_stop(self, results: list[ProfilingResultEvent]) -> bool:
        """Apply the same policy to a single-shape caller."""
        return self.should_stop_samples((results,))


class LatencyBudgetStrategy(ProfilingTuneStrategy):
    """Searches for max throughput while satisfying a latency budget.

    Each backend is profiled across the configured batch-size sweep. Both shape
    latencies must be at most ``latency_budget_ms`` before a batch is scored.
    Eligible batches are ranked by geometric-mean throughput; min-shape throughput
    and latency remain the reported values.

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
        profiling_results = profile_backend(
            backend,
            name,
            graph_spec,
            samples,
            profiling_cfg,
            sample_preparer=partial(
                shape_boundary_samples, graph_spec=graph_spec, preserve_batch_axes=profiling_cfg.batching
            ),
        )
        if profiling_results.status != ProfilingStatus.Status.SUCCESS:
            raise profiling_results.error or RuntimeError("Profiling failed")

        candidate = _find_best_latency_budget_candidate(
            profiling_results.results,
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
        return result.selection_score > other.selection_score

    def _speedup(self, result: BackendProfilingResult, baseline_result: BackendProfilingResult) -> float:
        return result.metric / baseline_result.metric

    def _boundary_speedup(
        self, candidate_latency: float, eager_latency: float, *, candidate_batch_size: int, eager_batch_size: int
    ) -> float:
        return throughput_speedup(
            candidate_latency,
            eager_latency,
            candidate_batch_size=candidate_batch_size,
            eager_batch_size=eager_batch_size,
        )

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
    profiling_results: "ProfilingResults",
    measuring_stop_strategy: MeasuringStopStrategy,
    latency_budget_ms: float,
) -> LatencyBudgetProfilingResult | None:
    """Rank budget-compliant boundary pairs, retaining min-shape values for reporting."""
    best: LatencyBudgetProfilingResult | None = None
    for batch_size, sample_latencies in profiling_results.batch_latencies(measuring_stop_strategy).items():
        latencies = BoundaryLatencies.from_samples(sample_latencies)
        within_budget = max(latencies.min_shape, latencies.max_shape) <= latency_budget_ms
        if not within_budget:
            continue

        # The geometric mean cannot make an over-budget boundary eligible.
        # Calculate ranking scores only after both independent checks pass.
        candidate = LatencyBudgetProfilingResult(
            selected_batch_size=batch_size,
            throughput=batch_size * 1000 / latencies.min_shape,
            latency=latencies.min_shape,
            boundary_latencies=latencies,
        )
        if best is None or candidate.selection_score > best.selection_score:
            best = candidate

    return best
