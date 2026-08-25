# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Find max batch size for a model."""

import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn

from aitune.torch.backend import TorchEagerBackend
from aitune.torch.backend.backend import Backend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import Sample
from aitune.torch.task.profiling.config import ProfilingConfig
from aitune.torch.task.profiling.events import ProfilingResultEvent, get_inference_events
from aitune.torch.task.profiling.measuring_stop_strategy import MeasuringStopStrategy
from aitune.torch.task.profiling.metrics import get_throughput
from aitune.torch.task.profiling.profiling import ProfilingResults, ProfilingStatus, profile_backend
from aitune.utils.logging import control_output
from aitune.utils.monitoring import annotate

logger = logging.getLogger(__name__)


@annotate(color="purple")
def find_max_batch_size(
    module: nn.Module,
    name: str,
    graph_spec: GraphSpec,
    data: Sequence[Sample],
    profiling_config: ProfilingConfig,
    device: torch.device,
    cache_dir: Path,
) -> tuple[int, float, ProfilingResults]:
    """Finds max batch size for Torch Compile as a baseline.

    Uses profiling with max throughput strategy to find max batch size.

    Note: This function expects user to set profiling_config.max_batch_size to the highest batch size they want to profile.

    Args:
        module: Model to find max batch size for.
        name: Name of the model.
        graph_spec: Graph spec of the model.
        data: Data to profile.
        profiling_config: Profiling configuration.
        torch_backend: Backend to use for the find max batch size. If not provided, Torch Eager backend will be used.
        device: Device to use for the calculation.
        cache_dir: Cache directory to store the backend artifacts.
    """
    backend = TorchEagerBackend()
    backend_cache_dir = cache_dir / backend.key()
    log_file = _log_file(backend_cache_dir, "build.log")
    with control_output(log_file=log_file):
        backend.build(module, graph_spec, data, device, backend_cache_dir)
    return find_max_throughput_for_backend(backend, name, graph_spec, data, profiling_config)


def find_max_throughput_for_backend(
    backend: Backend,
    name: str,
    graph_spec: GraphSpec,
    data: Sequence[Sample],
    profiling_config: ProfilingConfig,
) -> tuple[int, float, ProfilingResults]:
    """Profiles a backend to find the batch size that achieves maximum throughput.

    Args:
        module: Model to calculate maximum throughput for.
        name: Name of the model.
        graph_spec: Graph spec of the model.
        data: Data to profile.
        profiling_config: Profiling configuration.
        backend: Backend to use for the calculation.
        device: Device to use for the calculation.

    Returns:
        Tuple containing:
            - Batch size with maximum throughput.
            - Throughput for the batch size.
            - Backend used for the calculation.
            - Profiling results.
    """
    profiling_results = profile_backend(
        backend,
        name,
        graph_spec,
        data,
        profiling_config,
    )

    if profiling_results.status != ProfilingStatus.Status.SUCCESS:
        logger.debug("Profiling failed for %s", backend.describe(), exc_info=profiling_results.error)
        raise profiling_results.error

    throughput_per_batch_size = get_throughput_per_batch_size(
        get_inference_events(profiling_results.results.entries), profiling_config.measurement_stop_strategy
    )

    if len(throughput_per_batch_size) == 0:
        raise ValueError(f"No throughput data found for backend {backend.describe()}")

    batch_size, throughput = throughput_per_batch_size[0]
    return batch_size, throughput, profiling_results.results


def get_throughput_per_batch_size(
    profiling_results: list[ProfilingResultEvent], measuring_stop_strategy: MeasuringStopStrategy
) -> list[tuple[int, float]]:
    """Gets throughput per batch size."""
    events_batch_size = defaultdict(list)
    for event in profiling_results:
        events_batch_size[event.batch_size].append(event)

    throughput_per_batch_size = []
    for batch_size, events in events_batch_size.items():
        measured_events = measuring_stop_strategy.get_events(events)
        throughput = get_throughput(measured_events, batch_size)
        throughput_per_batch_size.append((batch_size, throughput))

    return sorted(throughput_per_batch_size, key=lambda x: x[1], reverse=True)


def _log_file(cache_dir: Path, filename: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_file = cache_dir / filename
    return log_file
