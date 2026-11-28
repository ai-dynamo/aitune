# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profiling utilities for AITune models.

TODO: add support for LLM token profiling, strategies for LLM benchmarking
TODO: add support for additional out of module(this source) events

"""

import logging
from collections import defaultdict
from collections.abc import Callable, Generator, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from aitune.torch.backend.backend import Backend
from aitune.torch.dataloader import DataLoaderFactory, DatasetLike, samples_generator
from aitune.torch.distributed import coordinator
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import Sample, SampleStore
from aitune.torch.task.profiling.config import ProfilingConfig
from aitune.torch.task.profiling.events import ProfilingResultEvent, get_inference_events
from aitune.torch.task.profiling.measuring_stop_strategy import MeasuringStopStrategy
from aitune.torch.task.profiling.metrics import get_latency

logger = logging.getLogger(__name__)


@dataclass
class BatchProfilingResults:
    """Measurements of one or more samples at one batch size, in caller-defined order.

    The first sample is primary for reporting and default saturation decisions.
    The profiler does not assign shape meanings to the remaining samples.
    """

    batch_size: int
    sample_events: list[list[ProfilingResultEvent]]

    def latencies(self, measuring_stop_strategy: MeasuringStopStrategy) -> tuple[float, ...]:
        """Reduce each sample independently using its own warmup or stability window."""
        return tuple(
            get_latency(measuring_stop_strategy.get_events(get_inference_events(events)))
            for events in self.sample_events
        )


@dataclass
class ProfilingResults:
    """Flat events for existing callers, with complete per-batch sample groups.

    entries preserves the original flat result API. batches retains the grouping
    needed by callers that explicitly profile multiple samples per batch.
    """

    entries: list[ProfilingResultEvent] = field(default_factory=list)
    batches: list[BatchProfilingResults] = field(default_factory=list)

    def add_batch(self, batch: BatchProfilingResults) -> None:
        """Retain a complete batch in both the flat and grouped result views."""
        self.batches.append(batch)
        for events in batch.sample_events:
            self.entries.extend(events)

    def batch_latencies(self, measuring_stop_strategy: MeasuringStopStrategy) -> dict[int, tuple[float, ...]]:
        """Return sample means for each batch, supporting original flat-only results."""
        if self.batches:
            return {batch.batch_size: batch.latencies(measuring_stop_strategy) for batch in self.batches}

        # Original callers can still construct ProfilingResults(entries=...).
        events_by_batch = defaultdict(list)
        for event in self.entries:
            events_by_batch[event.batch_size].append(event)
        return {
            batch: BatchProfilingResults(batch, [events]).latencies(measuring_stop_strategy)
            for batch, events in events_by_batch.items()
        }


@dataclass
class ProfilingStatus:
    """Profiling outcome and any complete batches collected before it finished."""

    class Status(Enum):
        """Profiling status."""

        SUCCESS = "success"
        FAILURE = "failure"

    status: Status

    error: Any | None = None
    results: ProfilingResults = field(default_factory=ProfilingResults)


def profile(
    func: Callable,
    dataset: DatasetLike | DataLoaderFactory | torch.Tensor,
    profiling_config: ProfilingConfig,
    model_name: str | None = None,
    backend_details: str | None = None,
) -> ProfilingStatus:
    """Profile a callable with given dataset.

    Args:
        func: Function to profile.
        dataset: Dataset to profile.
        profiling_config: Configuration for profiling.
        model_name: Name of the model, optional.
        backend_details: Detailed information of the backend, optional.

    Returns:
        ProfilingStatus: Status of the profiling.
    """
    samples = samples_generator(dataset, profiling_config.batch_sizes, max_num_batches_per_batch_size=1)
    batches = ((batch_size, ((args, kwargs),)) for batch_size, args, kwargs in samples)
    return _profile_samples(
        func,
        batches,
        profiling_config,
        model_name=model_name,
        backend_details=backend_details,
    )


def profile_backend(
    backend: Backend,
    name: str,
    graph_spec: GraphSpec,
    samples: SampleStore,
    profiling_config: ProfilingConfig,
    *,
    sample_preparer: Callable[[Sample], Sequence[Sample]] | None = None,
) -> ProfilingStatus:
    """Profile a backend using one recorded sample unless preparation is requested.

    Args:
        backend: Built backend to measure.
        name: Model name used in events and errors.
        graph_spec: Graph metadata used to construct each requested batch.
        samples: Recorded samples; only the first is loaded.
        profiling_config: Batch sweep and measurement policies.
        sample_preparer: Optional function that expands the recorded sample into
            one or more samples. Their order defines the result order; the first
            is primary for reporting and default saturation. For example, a tuning
            strategy can explicitly prepare min/max inputs with minimum first.

    Returns:
        Status, flat events, and complete per-batch sample measurements.

    Note:
        Recorded-sample loading, preparation, and batch construction happen before
        timing starts. The default keeps the original single-sample behavior.
    """

    def generator():
        with coordinator.raise_if_any_rank_fails(f"Preparing profiling samples for {name}"):
            sample = samples[0]
            prepared_samples = tuple(sample_preparer(sample)) if sample_preparer is not None else (sample,)
            if not prepared_samples:
                raise ValueError("Profiling requires at least one sample per batch")
        if profiling_config.batching:
            for batch_size in profiling_config.batch_sizes:
                with coordinator.raise_if_any_rank_fails(f"Preparing profiling batch {batch_size} for {name}"):
                    batch_samples = tuple(graph_spec.make_batch(*sample, batch_size) for sample in prepared_samples)
                yield batch_size, batch_samples
        else:
            yield 1, prepared_samples

    return _profile_samples(
        backend.infer, generator(), profiling_config, model_name=name, backend_details=backend.describe()
    )


def _profile_samples(
    func: Callable,
    batches: Generator[tuple[int, Sequence[tuple[list | tuple, dict]]], None, None],
    profiling_config: ProfilingConfig,
    model_name: str | None = None,
    backend_details: str | None = None,
) -> ProfilingStatus:
    """Measure every supplied sample before recording a batch and deciding whether to stop."""
    profiling_stop_strategy = deepcopy(profiling_config.profiling_stop_strategy)
    profiling_results = ProfilingResults()
    try:
        for batch_size, samples in batches:
            logger.debug("Profiling %s with backend %s and batch size %d", model_name, backend_details, batch_size)
            coordinator.verify_equal(len(samples), "profiling samples per batch")
            batch_results = _measure_batch_samples(
                batch_size,
                func,
                samples,
                profiling_config,
                model_name=model_name or "model",
                backend_details=backend_details or "backend",
            )
            if batch_results is None:
                break
            profiling_results.add_batch(batch_results)

            local_should_stop = False
            with coordinator.raise_if_any_rank_fails("Profiling stop decision"):
                local_should_stop = profiling_stop_strategy.should_stop_samples([
                    get_inference_events(events) for events in batch_results.sample_events
                ])
            if coordinator.decide_by_rank0(local_should_stop):
                break
        return ProfilingStatus(status=ProfilingStatus.Status.SUCCESS, results=profiling_results)
    except Exception as error:
        return ProfilingStatus(status=ProfilingStatus.Status.FAILURE, error=error, results=profiling_results)


def _measure_batch_samples(
    batch_size: int,
    model: Callable,
    samples: Sequence[tuple[list | tuple, dict]],
    profiling_config: ProfilingConfig,
    model_name: str,
    backend_details: str,
) -> BatchProfilingResults | None:
    """Return a complete measurement group, discarding the batch if any sample fails."""
    if not samples:
        raise ValueError("Profiling requires at least one sample per batch")
    sample_events = []
    for sample in samples:
        events = _run_profiling_batch_size(batch_size, model, sample, profiling_config, model_name, backend_details)
        if events is None:
            return None
        sample_events.append(events)
    return BatchProfilingResults(batch_size, sample_events)


def _run_profiling_batch_size(
    batch_size: int,
    model: Callable,
    sample: tuple[list | tuple, dict],
    profiling_config: ProfilingConfig,
    model_name: str,
    backend_details: str,
) -> list[ProfilingResultEvent] | None:
    """Measure one shape until its measurement policy stops, including warmup events.

    Copying the policy isolates counters and windows for every boundary. All ranks agree on
    measurement completion or failure before entering the next timing iteration.
    """
    entries = []
    measurement_stop_strategy = deepcopy(profiling_config.measurement_stop_strategy)
    coordinator.barrier()
    while True:
        local_error: Exception | None = None
        local_should_stop = False
        new_entries: list[ProfilingResultEvent] | None = None
        try:
            # make call idempotent (no side effects) since model can be stateful like LLM KV cache
            new_entries = profiling_config.measuring_strategy.do_measurement(
                batch_size, model, deepcopy(sample), backend_details=backend_details, model_name=model_name
            )
            local_should_stop = measurement_stop_strategy.should_stop(get_inference_events(new_entries))
        except torch.OutOfMemoryError as error:
            local_error = error
            logger.warning(
                "Out of memory error while profiling %s with backend %s and batch size %d",
                model_name,
                backend_details,
                batch_size,
            )
        except Exception as error:
            local_error = error
            logger.warning(
                "Error while profiling %s with backend %s and batch size %d: %s",
                model_name,
                backend_details,
                batch_size,
                error,
            )

        if not coordinator.outcome(local_error).succeeded:
            return None

        if new_entries is None:
            raise RuntimeError("Profiling succeeded on all ranks but produced no entries")
        entries += new_entries
        if coordinator.decide_by_rank0(local_should_stop):
            break

    # Note: keeping and returning all entries
    return entries
