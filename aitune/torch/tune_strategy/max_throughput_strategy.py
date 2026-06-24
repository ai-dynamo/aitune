# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Max throughput tune strategy.

1. Finds max batch size.
2. Profiles TorchEager as a throughput baseline when performance validation is enabled.
3. Runs all user-provided backends with the same sweep.
4. Returns the backend with max throughput; falls back to TorchEager when
   performance validation is enabled and no user backend beats the baseline.
"""

from aitune.torch.backend import Backend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.task.find_max_batch_size import find_max_throughput_for_backend
from aitune.torch.task.profiling import ProfilingConfig
from aitune.torch.tune_strategy.profiling_tune_strategy import ProfilingTuneStrategy


class MaxThroughputStrategy(ProfilingTuneStrategy):
    """Searches and selects the backend with max throughput.

    TorchEager is profiled in _pre_tune as a throughput baseline when performance validation is enabled
    (not injected into the backends list). When validation is enabled, the strategy falls back to
    TorchEager when no user-provided backend beats it. When disabled, the best user-provided backend
    wins and the strategy raises if all user backends fail.
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
        data: list[Sample],
        profiling_cfg: ProfilingConfig,
    ) -> tuple[int, float]:
        """Profiles the backend and returns (batch_size, throughput)."""
        batch_size, throughput, _ = find_max_throughput_for_backend(backend, name, graph_spec, data, profiling_cfg)
        return batch_size, throughput

    def _is_better(self, value: float, other: float) -> bool:
        return value > other

    def _speedup(self, value: float, baseline_value: float) -> float:
        return value / baseline_value
