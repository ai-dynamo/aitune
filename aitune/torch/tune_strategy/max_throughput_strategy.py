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

from aitune.torch.backend import Backend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import SampleStore
from aitune.torch.task.find_max_batch_size import find_max_throughput_for_backend
from aitune.torch.task.profiling import ProfilingConfig
from aitune.torch.task.profiling.events import get_inference_events
from aitune.torch.task.profiling.metrics import get_latency
from aitune.torch.tune_strategy.profiling_tune_strategy import BackendProfilingResult, ProfilingTuneStrategy


@dataclass(kw_only=True)
class MaxThroughputProfilingResult(BackendProfilingResult):
    """Profiling result for max-throughput selection."""

    selected_batch_size: int
    throughput: float
    latency: float

    @property
    def metric(self) -> float:
        """Returns throughput as the comparison metric."""
        return self.throughput

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
        """Profiles the backend and returns throughput with the selected batch size."""
        batch_size, throughput, profiling_results = find_max_throughput_for_backend(
            backend, name, graph_spec, samples, profiling_cfg
        )
        selected_events = [
            event for event in get_inference_events(profiling_results.entries) if event.batch_size == batch_size
        ]
        measured_events = profiling_cfg.measurement_stop_strategy.get_events(selected_events)
        if not measured_events:
            raise ValueError(f"No latency measurements found for {backend.describe()} at batch size {batch_size}")
        return MaxThroughputProfilingResult(
            throughput=throughput, latency=get_latency(measured_events), selected_batch_size=batch_size
        )

    def _is_better(self, result: BackendProfilingResult, other: BackendProfilingResult) -> bool:
        return result.metric > other.metric

    def _speedup(self, result: BackendProfilingResult, baseline_result: BackendProfilingResult) -> float:
        return result.metric / baseline_result.metric
