# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Find max batch size for a model."""

import logging
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from aitune.torch.backend.backend import Backend
from aitune.torch.distributed import coordinator
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import SampleStore
from aitune.torch.task.profiling.config import ProfilingConfig
from aitune.torch.task.profiling.events import ProfilingResultEvent, get_inference_events
from aitune.torch.task.profiling.measuring_stop_strategy import MeasuringStopStrategy
from aitune.torch.task.profiling.metrics import aggregate_throughput_per_batch_size, get_throughput
from aitune.torch.task.profiling.profiling import ProfilingResults, ProfilingStatus, profile_backend
from aitune.torch.utils.module import get_default_backend_for_module
from aitune.utils.logging import control_output
from aitune.utils.monitoring import annotate

logger = logging.getLogger(__name__)


@annotate(color="purple")
def find_max_batch_size(
    module: nn.Module,
    name: str,
    graph_spec: GraphSpec,
    samples: SampleStore,
    profiling_config: ProfilingConfig,
    device: torch.device,
    cache_dir: Path,
) -> tuple[int, float, ProfilingResults]:
    """Find the highest-scoring batch using the module's default backend.

    The search uses profiling_config.batch_sizes and its profiling stop policy.
    The selected batch maximizes min-shape throughput, rather than the
    largest batch that fits in memory.

    Args:
        module: Model to find max batch size for.
        name: Name of the model.
        graph_spec: Graph spec of the model.
        samples: Recorded samples to profile.
        profiling_config: Profiling configuration.
        device: Device to use for the calculation.
        cache_dir: Cache directory to store the backend artifacts.

    Returns:
        Selected batch size, min-shape throughput, and raw profiling results.
    """
    backend = get_default_backend_for_module(module)
    backend_cache_dir = cache_dir / backend.key()
    log_file = _log_file(backend_cache_dir, "build.log")
    with coordinator.raise_if_any_rank_fails("Building find-max-batch-size backend"):
        with control_output(log_file=log_file):
            backend.build(module, graph_spec, samples, device, backend_cache_dir)
    return find_max_throughput_for_backend(backend, name, graph_spec, samples, profiling_config)


def find_max_throughput_for_backend(
    backend: Backend,
    name: str,
    graph_spec: GraphSpec,
    samples: SampleStore,
    profiling_config: ProfilingConfig,
) -> tuple[int, float, ProfilingResults]:
    """Select a batch using min-shape throughput, preserving the discovery policy.

    This helper also serves callers outside MaxThroughputStrategy. Combined
    boundary scoring belongs to that strategy, rather than changing discovery's
    batch choice. Ranks coordinate failures and must expose matching batch sets.

    Args:
        backend: Built backend to profile.
        name: Name of the model.
        graph_spec: Graph spec of the model.
        samples: Recorded samples to profile.
        profiling_config: Profiling configuration.

    Returns:
        Selected batch size, its min-shape throughput across ranks, and the local
        raw boundary measurements used by the strategy's result construction.
    """
    profiling_results = profile_backend(
        backend,
        name,
        graph_spec,
        samples,
        profiling_config,
    )

    local_error: Exception | None = None
    throughput_per_batch_size = None
    try:
        if profiling_results.status != ProfilingStatus.Status.SUCCESS:
            logger.debug("Profiling failed for %s", backend.describe(), exc_info=profiling_results.error)
            raise profiling_results.error

        throughput_per_batch_size = get_throughput_per_batch_size(
            get_inference_events(profiling_results.results.entries), profiling_config.measurement_stop_strategy
        )
    except Exception as error:
        local_error = error

    outcome, gathered_throughputs = coordinator.collect_results(throughput_per_batch_size, local_error)
    outcome.raise_if_failed(f"Processing profiling results for {backend.describe()}", local_error)
    if throughput_per_batch_size is None:
        raise RuntimeError("Profiling succeeded on all ranks but produced no throughput results")
    throughput_per_batch_size = aggregate_throughput_per_batch_size([
        dict(candidate) for candidate in gathered_throughputs
    ])

    if not throughput_per_batch_size:
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
