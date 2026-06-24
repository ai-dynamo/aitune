# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Min latency tune strategy."""

from dataclasses import replace

from aitune.torch.backend import Backend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.task.profiling import ProfilingConfig
from aitune.torch.task.profiling.events import get_inference_events
from aitune.torch.task.profiling.metrics import get_latency
from aitune.torch.task.profiling.profiling import ProfilingStatus, profile_backend
from aitune.torch.tune_strategy.profiling_tune_strategy import ProfilingTuneStrategy


class MinLatencyStrategy(ProfilingTuneStrategy):
    """Searches and selects the backend with minimum latency at batch size 1.

    TorchEager is profiled in _pre_tune as a latency baseline when baseline validation is enabled
    (not injected into the backends list). When validation is enabled, the strategy falls back to
    TorchEager when no user-provided backend beats it. When disabled, the best user-provided backend
    wins and the strategy raises if all user backends fail.
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
        data: list[Sample],
        profiling_cfg: ProfilingConfig,
    ) -> tuple[int, float]:
        """Profiles the backend at batch size 1 and returns (1, latency_ms)."""
        profiling_results = profile_backend(backend, name, graph_spec, data, profiling_cfg)
        if profiling_results.status != ProfilingStatus.Status.SUCCESS:
            raise profiling_results.error or RuntimeError("Profiling failed")
        events = profiling_cfg.measurement_stop_strategy.get_events(
            get_inference_events(profiling_results.results.entries)
        )
        return 1, get_latency(events)

    def _get_profiling_config(self, batching: bool, max_batch_size: int) -> ProfilingConfig:
        return replace(self.profiling_config, batch_sizes=[1], batching=False)

    def _is_better(self, value: float, other: float) -> bool:
        return value < other

    def _speedup(self, value: float, baseline_value: float) -> float:
        return baseline_value / value
