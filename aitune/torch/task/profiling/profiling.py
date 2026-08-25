# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profiling utilities for AITune models.

TODO: add support for LLM token profiling, strategies for LLM benchmarking
TODO: add support for additional out of module(this source) events

"""

import logging
from collections.abc import Callable, Generator, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from aitune.torch.backend.backend import Backend
from aitune.torch.dataloader import DataLoaderFactory, DatasetLike, samples_generator
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import Sample
from aitune.torch.task.profiling.config import ProfilingConfig
from aitune.torch.task.profiling.events import ProfilingResultEvent, get_inference_events

logger = logging.getLogger(__name__)


@dataclass
class ProfilingResults:
    """Profiling results."""

    entries: list[ProfilingResultEvent] = field(default_factory=list)


@dataclass
class ProfilingStatus:
    """Profiling status."""

    class Status(Enum):
        """Profiling status."""

        SUCCESS = "success"
        FAILURE = "failure"

    status: Status

    error: Any | None = None
    results: ProfilingResults | None = None


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
    return _profile(
        func,
        samples_generator(dataset, profiling_config.batch_sizes, max_num_batches_per_batch_size=1),
        profiling_config,
        model_name=model_name,
        backend_details=backend_details,
    )


def profile_backend(
    backend: Backend,
    name: str,
    graph_spec: GraphSpec,
    data: Sequence[Sample],
    profiling_config: ProfilingConfig,
) -> ProfilingStatus:
    """Profile a backend with graph spec and data.

    Args:
        backend: Backend to profile.
        name: Name of the model.
        graph_spec: Graph spec of the model. Used to generate batch samples.
        data: Data to profile. Only first sample is used to generate batch samples.
        profiling_config: Profiling configuration.

    Returns:
        ProfilingStatus: Status of the profiling.

    Note:
        The first recorded sample is loaded and expanded to each requested batch
        size before :class:`ModelExecutionTimeMeasuringStrategy` starts its timer.
        This keeps ``SampleStore`` disk I/O and batch construction out of reported
        inference latency and throughput.
    """

    def generator():
        base_args, base_kwargs = data[0]
        if profiling_config.batching:
            for batch_size in profiling_config.batch_sizes:
                args, kwargs = graph_spec.make_batch(base_args, base_kwargs, batch_size)
                yield batch_size, args, kwargs
        else:
            yield 1, base_args, base_kwargs

    return _profile(
        backend.infer,
        generator(),
        profiling_config,
        model_name=name,
        backend_details=backend.describe(),
    )


def _profile(
    func: Callable,
    samples: Generator[tuple[int, list | tuple, dict], None, None],
    profiling_config: ProfilingConfig,
    model_name: str | None = None,
    backend_details: str | None = None,
):
    """Profile a model."""
    profiling_stop_strategy = deepcopy(profiling_config.profiling_stop_strategy)
    profiling_results = ProfilingResults()
    try:
        for batch_size, args, kwargs in samples:
            logger.debug("Profiling %s with backend %s and batch size %d", model_name, backend_details, batch_size)
            new_entries = _run_profiling_batch_size(
                batch_size,
                func,
                (args, kwargs),
                profiling_config=profiling_config,
                model_name=model_name or "model",
                backend_details=backend_details or "backend",
            )

            if new_entries is None:
                break

            profiling_results.entries += new_entries

            if profiling_stop_strategy.should_stop(get_inference_events(new_entries)):
                break

        return ProfilingStatus(status=ProfilingStatus.Status.SUCCESS, results=profiling_results)

    except Exception as e:
        return ProfilingStatus(status=ProfilingStatus.Status.FAILURE, error=e, results=profiling_results)


def _run_profiling_batch_size(
    batch_size: int,
    model: Callable,
    sample: tuple[list | tuple, dict],
    profiling_config: ProfilingConfig,
    model_name: str,
    backend_details: str,
) -> list[ProfilingResultEvent] | None:
    entries = []
    measurement_stop_strategy = deepcopy(profiling_config.measurement_stop_strategy)
    try:
        while True:
            # make call idempotent (no side effects) since model can be stateful like LLM KV cache
            new_entries = profiling_config.measuring_strategy.do_measurement(
                batch_size, model, deepcopy(sample), backend_details=backend_details, model_name=model_name
            )
            entries += new_entries

            if measurement_stop_strategy.should_stop(get_inference_events(new_entries)):
                break
    except torch.OutOfMemoryError:
        logger.warning(
            "Out of memory error while profiling %s with backend %s and batch size %d",
            model_name,
            backend_details,
            batch_size,
        )
        return None
    except Exception as e:
        logger.warning(
            "Error while profiling %s with backend %s and batch size %d: %s",
            model_name,
            backend_details,
            batch_size,
            e,
        )
        return None

    # Note: keeping and returning all entries
    return entries
