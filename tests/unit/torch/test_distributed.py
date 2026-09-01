# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for non-owning distributed runtime integration."""

import ast
import errno
import os
import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from aitune.torch.distributed import (
    DistributedContext,
    DistributedCoordinator,
    DistributedOutcome,
    distributed_cache_dir,
    distributed_context,
    distributed_output_path,
    resolve_tuning_device,
)
from aitune.torch.distributed import (
    coordinator as shared_coordinator,
)


def _distributed_worker(rank: int, world_size: int, rendezvous_path: str, result_dir: str) -> None:
    """Application-owned process-group setup used by the multiprocess integration test."""
    interfaces = {name for _, name in socket.if_nameindex()}
    os.environ["GLOO_SOCKET_IFNAME"] = "lo" if "lo" in interfaces else "lo0"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        context = distributed_context()
        coordinator = DistributedCoordinator(context)
        outcome = coordinator.outcome(None if rank == 0 else RuntimeError("rank failure"))
        inspection_order = coordinator.broadcast_from_rank0(("encoder", "decoder") if rank == 0 else None)
        result = (
            context.rank,
            context.world_size,
            outcome.succeeded,
            outcome.failed_ranks,
            coordinator.collect(float(rank + 1)),
            inspection_order,
        )
        coordinator.barrier()
        (Path(result_dir) / f"rank-{rank}.txt").write_text(repr(result))
    finally:
        # The test application owns teardown; AITune never performs it.
        dist.destroy_process_group()


def test_detects_torchrun_environment_without_initializing_distributed(mocker):
    mocker.patch.dict("os.environ", {"RANK": "3", "LOCAL_RANK": "1", "WORLD_SIZE": "8"}, clear=True)
    mocker.patch("aitune.torch.distributed.dist.is_available", return_value=True)
    mocker.patch("aitune.torch.distributed.dist.is_initialized", return_value=False)
    init = mocker.patch("aitune.torch.distributed.dist.init_process_group")
    destroy = mocker.patch("aitune.torch.distributed.dist.destroy_process_group")

    context = distributed_context()

    assert context == DistributedContext(rank=3, local_rank=1, world_size=8, torch_distributed_initialized=False)
    init.assert_not_called()
    destroy.assert_not_called()


def test_detects_open_mpi_environment(mocker):
    mocker.patch.dict(
        "os.environ",
        {
            "OMPI_COMM_WORLD_RANK": "2",
            "OMPI_COMM_WORLD_LOCAL_RANK": "0",
            "OMPI_COMM_WORLD_SIZE": "4",
        },
        clear=True,
    )
    mocker.patch("aitune.torch.distributed.dist.is_available", return_value=False)

    context = distributed_context()

    assert (context.rank, context.local_rank, context.world_size) == (2, 0, 4)


def test_initialized_process_group_is_authoritative(mocker):
    mocker.patch.dict("os.environ", {"RANK": "7", "LOCAL_RANK": "1", "WORLD_SIZE": "8"}, clear=True)
    mocker.patch("aitune.torch.distributed.dist.is_available", return_value=True)
    mocker.patch("aitune.torch.distributed.dist.is_initialized", return_value=True)
    mocker.patch("aitune.torch.distributed.dist.get_rank", return_value=2)
    mocker.patch("aitune.torch.distributed.dist.get_world_size", return_value=4)

    context = distributed_context()

    assert context == DistributedContext(rank=2, local_rank=1, world_size=4, torch_distributed_initialized=True)


def test_multi_process_coordination_requires_existing_process_group():
    context = DistributedContext(rank=0, local_rank=0, world_size=2, torch_distributed_initialized=False)

    with pytest.raises(RuntimeError, match="application-initialized"):
        DistributedCoordinator(context)


def test_shared_coordinator_detects_context_lazily(mocker):
    context = DistributedContext(rank=0, local_rank=0, world_size=1, torch_distributed_initialized=False)
    detect = mocker.patch("aitune.torch.distributed.distributed_context", return_value=context)

    assert shared_coordinator.collect("value") == ["value"]
    detect.assert_called_once_with()


def test_coordinator_collects_distributed_outcome(mocker):
    context = DistributedContext(rank=0, local_rank=0, world_size=2, torch_distributed_initialized=True)

    def gather(values, value):
        values[:] = [value, "RuntimeError"]

    gather_mock = mocker.patch("aitune.torch.distributed.dist.all_gather_object", side_effect=gather)

    outcome = DistributedCoordinator(context).outcome(None)

    assert not outcome.succeeded
    assert outcome.failed_ranks == (1,)
    with pytest.raises(RuntimeError, match=r"Profiling failed on ranks \[1\]"):
        outcome.raise_if_failed("Profiling")
    gather_mock.assert_called_once_with([None, "RuntimeError"], None)


def test_coordinator_collects_bounded_error_token(mocker):
    context = DistributedContext(rank=0, local_rank=0, world_size=2, torch_distributed_initialized=True)

    def gather(values, value):
        values[:] = [value, None]

    gather_mock = mocker.patch("aitune.torch.distributed.dist.all_gather_object", side_effect=gather)

    outcome = DistributedCoordinator(context).outcome(RuntimeError("x" * 100_000))

    assert outcome.errors == ("RuntimeError", None)
    gather_mock.assert_called_once_with(["RuntimeError", None], "RuntimeError")


def test_distributed_outcome_reports_out_of_space():
    outcome = DistributedCoordinator(DistributedContext()).outcome(OSError(errno.ENOSPC, "cache full"))

    assert outcome.out_of_space


def test_distributed_outcome_prioritizes_remote_out_of_space():
    outcome = DistributedOutcome(("ValueError", "OSError[ENOSPC]"))

    with pytest.raises(OSError) as raised:
        outcome.raise_if_failed("Tuning", ValueError("local failure"))

    assert raised.value.errno == errno.ENOSPC


def test_coordinator_collects_results_and_errors_together(mocker):
    context = DistributedContext(rank=0, local_rank=0, world_size=2, torch_distributed_initialized=True)

    def gather(values, value):
        values[:] = [value, ("RuntimeError", None)]

    gather_mock = mocker.patch("aitune.torch.distributed.dist.all_gather_object", side_effect=gather)

    outcome, results = DistributedCoordinator(context).collect_results({"throughput": 1.0}, None)

    assert not outcome.succeeded
    assert outcome.failed_ranks == (1,)
    assert results == [{"throughput": 1.0}, None]
    gather_mock.assert_called_once_with(
        [(None, {"throughput": 1.0}), ("RuntimeError", None)],
        (None, {"throughput": 1.0}),
    )


def test_coordinator_broadcasts_rank_zero_decision(mocker):
    context = DistributedContext(rank=1, local_rank=1, world_size=2, torch_distributed_initialized=True)

    def broadcast(decision, src):
        assert src == 0
        decision[0] = True

    broadcast_mock = mocker.patch("aitune.torch.distributed.dist.broadcast_object_list", side_effect=broadcast)

    decision = DistributedCoordinator(context).decide_by_rank0(False)

    assert decision is True
    broadcast_mock.assert_called_once_with([True], src=0)


def test_coordinator_broadcasts_rank_zero_value(mocker):
    context = DistributedContext(rank=1, local_rank=1, world_size=2, torch_distributed_initialized=True)

    def broadcast(value, src):
        assert src == 0
        value[0] = ("encoder", "decoder")

    broadcast_mock = mocker.patch("aitune.torch.distributed.dist.broadcast_object_list", side_effect=broadcast)

    value = DistributedCoordinator(context).broadcast_from_rank0(("local",))

    assert value == ("encoder", "decoder")
    broadcast_mock.assert_called_once_with([("encoder", "decoder")], src=0)


def test_distributed_outcome_preserves_local_error():
    local_error = ValueError("local failure")
    outcome = DistributedOutcome(("ValueError", None))

    with pytest.raises(ValueError, match="local failure") as raised:
        outcome.raise_if_failed("Profiling", local_error)

    assert raised.value is local_error


def test_coordinator_failure_context_preserves_local_error():
    coordinator = DistributedCoordinator(DistributedContext())
    local_error = ValueError("local failure")

    with pytest.raises(ValueError, match="local failure") as raised:
        with coordinator.raise_if_any_rank_fails("Profiling"):
            raise local_error

    assert raised.value is local_error


def test_coordinator_failure_context_raises_for_remote_error(mocker):
    context = DistributedContext(rank=0, local_rank=0, world_size=2, torch_distributed_initialized=True)

    def gather(values, value):
        values[:] = [value, "RuntimeError"]

    mocker.patch("aitune.torch.distributed.dist.all_gather_object", side_effect=gather)

    with pytest.raises(ValueError, match=r"Graph break detection failed on ranks \[1\]"):
        with DistributedCoordinator(context).raise_if_any_rank_fails(
            "Graph break detection",
            error_type=ValueError,
        ):
            pass


def test_resolve_tuning_device_uses_local_rank_without_changing_cuda_device(mocker):
    context = DistributedContext(rank=3, local_rank=1, world_size=4, torch_distributed_initialized=True)
    module = nn.Linear(2, 2)
    mocker.patch("aitune.torch.distributed.torch.cuda.is_available", return_value=True)
    set_device = mocker.patch("aitune.torch.distributed.torch.cuda.set_device")

    device = resolve_tuning_device(None, module=module, context=context)

    assert device == torch.device("cuda:1")
    set_device.assert_not_called()


@pytest.mark.parametrize("world_size", [1, 4])
def test_resolve_tuning_device_requires_cuda_for_implicit_fallback(mocker, world_size):
    context = DistributedContext(rank=0, local_rank=0, world_size=world_size)
    module = nn.Linear(2, 2)
    mocker.patch("aitune.torch.distributed.torch.cuda.is_available", return_value=False)

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        resolve_tuning_device(None, module=module, context=context)


def test_resolve_tuning_device_allows_explicit_cpu(mocker):
    mocker.patch("aitune.torch.distributed.torch.cuda.is_available", return_value=False)

    assert resolve_tuning_device("cpu") == torch.device("cpu")


def test_resolve_tuning_device_preserves_distributed_module_placement(mocker):
    module = nn.Linear(2, 2)
    # Treat the module as distributed to verify that its application-managed placement overrides the requested device.
    mocker.patch("aitune.torch.distributed.is_distributed_module", return_value=True)

    device = resolve_tuning_device("cuda:1", module=module)

    assert device == next(module.parameters()).device


def test_distributed_paths_are_rank_isolated(tmp_path):
    context = DistributedContext(rank=2, local_rank=0, world_size=4, torch_distributed_initialized=False)

    cache = distributed_cache_dir(tmp_path / "cache", context)
    report = distributed_output_path(tmp_path / "report.json", context)

    assert cache == tmp_path / "cache" / "rank-2-of-4"
    assert report == tmp_path / "report.rank-2-of-4.json"


@pytest.mark.skipif(
    sys.platform == "darwin" or not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo multiprocess test is unsupported on this platform",
)
def test_coordinator_with_application_initialized_process_group(tmp_path):
    rendezvous_path = tmp_path / "rendezvous"
    mp.spawn(
        _distributed_worker,
        args=(2, str(rendezvous_path), str(tmp_path)),
        nprocs=2,
        join=True,
    )

    results = [ast.literal_eval((tmp_path / f"rank-{rank}.txt").read_text()) for rank in range(2)]

    assert results == [
        (0, 2, False, (1,), [1.0, 2.0], ("encoder", "decoder")),
        (1, 2, False, (1,), [1.0, 2.0], ("encoder", "decoder")),
    ]
