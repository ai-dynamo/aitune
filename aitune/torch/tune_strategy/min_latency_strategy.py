# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Min latency tune strategy."""

from dataclasses import dataclass, replace
from functools import partial

from aitune.torch.backend import Backend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import SampleStore
from aitune.torch.task.profiling import ProfilingConfig
from aitune.torch.task.profiling.metrics import BoundaryLatencies
from aitune.torch.task.profiling.profiling import ProfilingStatus, profile_backend
from aitune.torch.tune_strategy.profiling_tune_strategy import BackendProfilingResult, ProfilingTuneStrategy
from aitune.torch.utils.shape_samples import shape_boundary_samples


@dataclass(kw_only=True)
class MinLatencyProfilingResult(BackendProfilingResult):
    """Profiling result for min-latency selection."""

    selected_batch_size: int = 1
    latency: float

    @property
    def metric(self) -> float:
        """Return the min-shape latency used in existing reports and speedup logs."""
        return self.latency

    @property
    def selection_score(self) -> float:
        """Rank using both shapes while reporting min-shape latency."""
        if self.boundary_latencies is None:
            return self.metric
        return self.boundary_latencies.latency_score


class MinLatencyStrategy(ProfilingTuneStrategy):
    """Searches and selects the backend with minimum latency at batch size 1.

    TorchEager is profiled in _pre_tune as a latency baseline unless performance validation is disabled
    (not injected into the backends list). In enabled mode, the strategy falls back to TorchEager when no
    user-provided backend beats it. In diagnostic or disabled mode, the best user-provided backend wins,
    and the strategy raises if all user backends fail.
    """

    _title = "Min Latency Strategy"
    _description = "evaluate all backends, return backend with min latency"
    _metric_label = "latency"
    _metric_unit = "ms"
    _value_fmt = ".3f"

    def __init__(self, *args, **kwargs):
        """Initializes MinLatencyStrategy with find_max_batch_size disabled."""
        super().__init__(*args, **kwargs)
        self.enable_find_max_batch_size(False)

    def _measure(
        self,
        backend: Backend,
        name: str,
        graph_spec: GraphSpec,
        samples: SampleStore,
        profiling_cfg: ProfilingConfig,
    ) -> MinLatencyProfilingResult:
        """Profiles the backend at batch size 1 and returns latency."""
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
        latencies = BoundaryLatencies.from_samples(
            profiling_results.results.batch_latencies(profiling_cfg.measurement_stop_strategy)[1]
        )
        return MinLatencyProfilingResult(latency=latencies.min_shape, boundary_latencies=latencies)

    def _get_profiling_config(self, batching: bool, max_batch_size: int) -> ProfilingConfig:
        return replace(self.profiling_config, batch_sizes=[1], batching=True)

    def _is_better(self, result: BackendProfilingResult, other: BackendProfilingResult) -> bool:
        return result.selection_score < other.selection_score

    def _speedup(self, result: BackendProfilingResult, baseline_result: BackendProfilingResult) -> float:
        return baseline_result.metric / result.metric

    def _boundary_speedup(
        self, candidate_latency: float, eager_latency: float, *, candidate_batch_size: int, eager_batch_size: int
    ) -> float:
        # Min-latency profiling compares the duration of batch-1 calls.
        return eager_latency / candidate_latency
