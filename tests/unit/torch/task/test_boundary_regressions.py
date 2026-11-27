# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from datetime import timedelta
from typing import Literal
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aitune.torch.backend import TorchEagerBackend
from aitune.torch.task.profiling import ProfilingConfig
from aitune.torch.task.profiling.measuring_stop_strategy import NumStepsMeasuringStopStrategy
from aitune.torch.task.profiling.profiling import ProfilingStatus, profile_backend
from tests.utilities.helpers import make_graph_spec, make_sample_store, update_input_spec


def _profile(
    tmp_path,
    first,
    second,
    *,
    batching,
    batch_size=1,
    recorded_batches=(1, 1),
    boundary: Literal["min_shape", "max_shape"] = "max_shape",
):
    model = torch.nn.Identity()
    sample = ((first,), {})
    graph_spec = make_graph_spec(model.forward, sample, batch_size=recorded_batches[0])
    update_input_spec(graph_spec, ((second,), {}), batch_size=recorded_batches[1])
    backend = Mock(spec=TorchEagerBackend)
    result = profile_backend(
        backend,
        "identity",
        graph_spec,
        make_sample_store([sample], tmp_path),
        ProfilingConfig(
            batch_sizes=[batch_size],
            batching=batching,
            shape_boundary=boundary,
            measurement_stop_strategy=NumStepsMeasuringStopStrategy(num_steps=1, warmup_samples=1),
        ),
    )
    return result, backend


def test_max_shape_sweep_keeps_recorded_rows_at_original_batch_size(tmp_path):
    recorded = torch.tensor([[1.0], [2.0]])
    result, backend = _profile(
        tmp_path, recorded, torch.ones(4, 3), batching=True, batch_size=2, recorded_batches=(2, 4)
    )
    assert result.status is ProfilingStatus.Status.SUCCESS
    assert torch.equal(backend.infer.call_args.args[0], recorded.repeat(1, 3))


def test_max_boundary_rejects_growth_from_an_empty_recorded_axis(tmp_path):
    result, backend = _profile(tmp_path, torch.empty(1, 0), torch.ones(1, 4), batching=False)
    assert result.status is ProfilingStatus.Status.FAILURE
    assert "empty axis" in str(result.error)
    assert "(1, 4)" in str(result.error)
    backend.infer.assert_not_called()


def test_min_boundary_can_keep_an_empty_axis(tmp_path):
    result, backend = _profile(tmp_path, torch.empty(1, 0), torch.ones(1, 4), batching=False, boundary="min_shape")

    assert result.status is ProfilingStatus.Status.SUCCESS
    assert backend.infer.call_args.args[0].shape == (1, 0)


def _rank_worker(rank, root, failing_operation):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / 'rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=5),
    )
    try:
        rank_dir = root / str(rank)
        rank_dir.mkdir()
        if rank == 1:
            with patch(
                failing_operation,
                side_effect=torch.OutOfMemoryError("rank-local allocation failed"),
            ):
                result, backend = _profile(rank_dir, torch.ones(1, 2), torch.ones(1, 4), batching=True)
        else:
            result, backend = _profile(rank_dir, torch.ones(1, 2), torch.ones(1, 4), batching=True)
        assert result.status is ProfilingStatus.Status.FAILURE
        backend.infer.assert_not_called()
        assert "rank" in str(result.error)
        assert "connection" not in str(result.error).lower()
        assert "timed out" not in str(result.error).lower()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    sys.platform == "darwin" or not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo multiprocess test is unsupported on this platform",
)
@pytest.mark.parametrize(
    "failing_operation",
    [
        "aitune.torch.task.profiling.profiling.resize_sample_to_shape",
        "aitune.torch.module.graph_spec.GraphSpec.make_batch",
    ],
)
def test_rank_local_preparation_failure_is_coordinated_before_inference(tmp_path, failing_operation):
    mp.spawn(_rank_worker, args=(tmp_path, failing_operation), nprocs=2, join=True)
