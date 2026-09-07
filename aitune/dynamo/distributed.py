# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collective request execution for Dynamo workers."""

import asyncio
import signal
from collections.abc import AsyncGenerator, Callable, Coroutine
from logging import getLogger
from typing import Any

import torch.distributed as dist

from aitune.dynamo.runtime import run_dynamo_worker

logger = getLogger(__name__)

_RUN_COMMAND = "run"
_STOP_COMMAND = "stop"


def should_run_collectively() -> bool:
    """Return whether the application-owned process group requires collective serving."""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def is_rank_zero() -> bool:
    """Return whether the current process owns the Dynamo endpoint and responses."""
    return not dist.is_initialized() or dist.get_rank() == 0


def run_collective_dynamo_worker(
    setup: Callable[[], None],
    serve: Callable[[Any], AsyncGenerator[Any, None]],
    namespace: str,
    component: str,
    endpoint: str,
    enable_nats: bool,
    on_ready: Callable[[Any, Any], Coroutine] | None,
    warmup: Callable[[], None] | None = None,
) -> None:
    """Serve one endpoint from rank zero while every rank executes each request."""
    _run_collective_setup(setup, stage="setup")
    if warmup is not None:
        _run_collective_setup(warmup, stage="warmup")
    if dist.get_rank() != 0:
        _run_collective_follower(serve)
        return

    worker_error: BaseException | None = None
    try:
        run_dynamo_worker(
            setup=lambda: None,
            serve=_collective_leader_serve(serve),
            namespace=namespace,
            component=component,
            endpoint=endpoint,
            enable_nats=enable_nats,
            on_ready=on_ready,
            warmup=None,
        )
    except BaseException as error:  # noqa: BLE001
        worker_error = error
        raise
    finally:
        try:
            _broadcast_collective_command(_STOP_COMMAND)
        except Exception:  # noqa: BLE001
            if worker_error is None:
                raise
            logger.exception("Failed to broadcast collective stop command while handling a worker error")


def _run_collective_setup(setup: Callable[[], None], stage: str = "setup") -> None:
    """Run a startup stage on every rank and prevent a partial worker from serving."""
    local_error = None
    try:
        setup()
    except Exception as error:  # noqa: BLE001
        local_error = error

    errors = _collective_errors(local_error)
    if any(errors):
        if local_error is not None:
            raise local_error
        raise RuntimeError(f"Collective Dynamo worker {stage} failed: " + "; ".join(filter(None, errors)))


def _collective_leader_serve(
    serve: Callable[[Any], AsyncGenerator[Any, None]],
) -> Callable[[Any], AsyncGenerator[Any, None]]:
    """Serialize rank-zero requests and execute each request on every rank."""
    request_lock = asyncio.Lock()

    async def collective_serve(request: Any) -> AsyncGenerator[Any, None]:
        async with request_lock:
            _broadcast_collective_command(_RUN_COMMAND, request)
            responses = await _execute_collective_request(serve, request)
            for response in responses:
                yield response

    return collective_serve


def _run_collective_follower(serve: Callable[[Any], AsyncGenerator[Any, None]]) -> None:
    """Receive rank-zero commands and execute requests without starting an endpoint."""
    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    try:
        # Rank zero converts an interactive interrupt into a coordinated stop command. Leave SIGTERM unchanged so a
        # supervisor can still terminate followers if rank zero exits before broadcasting that command.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        asyncio.run(_receive_collective_requests(serve))
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)


async def _receive_collective_requests(serve: Callable[[Any], AsyncGenerator[Any, None]]) -> None:
    """Execute rank-zero commands on a follower using one persistent event loop."""
    while True:
        command, request = _receive_collective_command()
        if command == _STOP_COMMAND:
            return
        if command != _RUN_COMMAND:
            raise RuntimeError(f"Unknown collective Dynamo worker command: {command!r}")
        try:
            await _execute_collective_request(serve, request, collect_responses=False)
        except Exception:
            logger.exception("Collective Dynamo request failed")


async def _execute_collective_request(
    serve: Callable[[Any], AsyncGenerator[Any, None]],
    request: Any,
    collect_responses: bool = True,
) -> list[Any]:
    """Execute one request locally and report rank-local failures collectively."""
    responses = []
    local_error = None
    try:
        async for response in serve(request):
            if collect_responses:
                responses.append(response)
    except Exception as error:  # noqa: BLE001
        local_error = error

    errors = _collective_errors(local_error)
    if any(errors):
        if local_error is not None:
            raise local_error
        raise RuntimeError("Collective Dynamo request failed: " + "; ".join(filter(None, errors)))
    return responses


def _collective_errors(error: Exception | None) -> list[str | None]:
    """Collect an error description from every worker rank."""
    local_error = None if error is None else f"rank {dist.get_rank()}: {type(error).__name__}: {error}"
    errors: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, local_error)
    return errors


def _broadcast_collective_command(command: str, request: Any = None) -> None:
    """Broadcast a request-control command from rank zero."""
    payload = [(command, request)]
    dist.broadcast_object_list(payload, src=0)


def _receive_collective_command() -> tuple[str, Any]:
    """Receive and validate one request-control command on a follower rank."""
    payload: list[tuple[str, Any] | None] = [None]
    dist.broadcast_object_list(payload, src=0)
    message = payload[0]
    if message is None:
        raise RuntimeError("Received an empty collective Dynamo worker command")
    return message
