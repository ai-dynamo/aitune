# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Max throughput tune strategy.

1. Finds max batch size.
2. Profiles TorchEager as a throughput baseline unless performance validation is disabled.
3. Runs all user-provided backends with the same sweep.
4. Returns the user backend with max throughput; in enabled mode, falls back to
   TorchEager when no user backend beats the baseline.
"""

from dataclasses import dataclass
from functools import partial

from aitune.torch.backend import Backend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import SampleStore
from aitune.torch.task.profiling import ProfilingConfig
from aitune.torch.task.profiling.metrics import (
    BoundaryLatencies,
    get_matching_batch_sizes,
    throughput_speedup,
)
from aitune.torch.tune_strategy.profiling_tune_strategy import BackendProfilingResult, ProfilingTuneStrategy
from aitune.torch.utils.shape_samples import shape_boundary_samples


@dataclass(kw_only=True)
class MaxThroughputProfilingResult(BackendProfilingResult):
    """Profiling result for max-throughput selection."""

    selected_batch_size: int
    throughput: float
    latency: float

    @property
    def metric(self) -> float:
        """Return the min-shape throughput used in existing reports and speedup logs."""
        return self.throughput

    @property
    def selection_score(self) -> float:
        """Rank using both shapes while reporting min-shape throughput."""
        if self.boundary_latencies is None:
            return self.metric
        return self.boundary_latencies.throughput_score(self.selected_batch_size)

    def to_json_dict(self, metric_label: str) -> dict[str, int | float]:
        """Return throughput and mean latency for the selected batch size."""
        result = super().to_json_dict(metric_label)
        result["latency"] = self.latency
        return result


class MaxThroughputStrategy(ProfilingTuneStrategy):
    """Searches and selects the backend with max throughput.

    TorchEager is profiled in _pre_tune as a throughput baseline unless performance validation is disabled
    (not injected into the backends list). In enabled mode, the strategy falls back to TorchEager when no
    user-provided backend beats it. In diagnostic or disabled mode, the best user-provided backend wins,
    and the strategy raises if all user backends fail.

    Callers either supply candidates explicitly or use ``resolve_strategy()`` to
    select and configure candidates for each module.
    """

    _title = "Max Throughput Strategy"
    _description = "evaluate all backends, return backend with max throughput"
    _metric_label = "throughput"
    _metric_unit = "samples/s"
    _value_fmt = ".2f"

    def _measure(
        self,
        backend: Backend,
        name: str,
        graph_spec: GraphSpec,
        samples: SampleStore,
        profiling_cfg: ProfilingConfig,
    ) -> MaxThroughputProfilingResult:
        """Choose a batch by combined boundary score, retaining min-shape report values."""
        profiling_status = profile_backend(
            backend,
            name,
            graph_spec,
            samples,
            profiling_cfg,
            sample_preparer=partial(
                shape_boundary_samples, graph_spec=graph_spec, preserve_batch_axes=profiling_cfg.batching
            ),
        )
        local_error: Exception | None = None
        latencies_by_batch: dict[int, BoundaryLatencies] = {}
        try:
            if profiling_status.status != ProfilingStatus.Status.SUCCESS:
                raise profiling_status.error or RuntimeError("Profiling failed")
            latencies_by_batch = {
                batch_size: BoundaryLatencies.from_samples(sample_latencies)
                for batch_size, sample_latencies in profiling_status.results.batch_latencies(
                    profiling_cfg.measurement_stop_strategy
                ).items()
            }
        except Exception as error:
            local_error = error

        outcome, gathered_latencies = coordinator.collect_results(latencies_by_batch, local_error)
        outcome.raise_if_failed(f"Processing profiling results for {backend.describe()}", local_error)
        batch_sizes = get_matching_batch_sizes(gathered_latencies)
        if not batch_sizes:
            raise ValueError(f"No throughput data found for backend {backend.describe()}")

        # Choose a batch using the same rank reduction as final backend comparison.
        batch_scores = {
            batch_size: BoundaryLatencies.worst_across_ranks([
                rank[batch_size] for rank in gathered_latencies
            ]).throughput_score(batch_size)
            for batch_size in batch_sizes
        }
        batch_size = min(batch_sizes, key=lambda size: (-batch_scores[size], size))
        latencies = latencies_by_batch[batch_size]
        return MaxThroughputProfilingResult(
            throughput=batch_size * 1000 / latencies.min_shape,
            latency=latencies.min_shape,
            selected_batch_size=batch_size,
            boundary_latencies=latencies,
        )

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
