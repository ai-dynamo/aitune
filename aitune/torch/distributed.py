# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-owning integration with an application-managed distributed runtime."""

import errno
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from aitune.torch.utils.module import get_module_device, is_distributed_module

_RANK_ENVIRONMENT_VARIABLES = ("RANK", "OMPI_COMM_WORLD_RANK")
_LOCAL_RANK_ENVIRONMENT_VARIABLES = ("LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK")
_WORLD_SIZE_ENVIRONMENT_VARIABLES = ("WORLD_SIZE", "OMPI_COMM_WORLD_SIZE")
_OUT_OF_SPACE_ERROR = "OSError[ENOSPC]"


@dataclass(frozen=True)
class DistributedContext:
    """A snapshot of distributed state owned by the calling application."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    torch_distributed_initialized: bool = False

    @property
    def is_multi_process(self) -> bool:
        """Return whether the launcher describes more than one process."""
        return self.world_size > 1

    @classmethod
    def _detect(cls) -> "DistributedContext":
        """Detect launcher and process-group state without modifying either."""
        initialized = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if initialized else cls._environment_integer(_RANK_ENVIRONMENT_VARIABLES, 0)
        world_size = (
            dist.get_world_size() if initialized else cls._environment_integer(_WORLD_SIZE_ENVIRONMENT_VARIABLES, 1)
        )
        local_rank = cls._environment_integer(_LOCAL_RANK_ENVIRONMENT_VARIABLES, -1)
        if local_rank < 0:
            local_rank = torch.cuda.current_device() if torch.cuda.is_available() else rank
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            torch_distributed_initialized=initialized,
        )

    @staticmethod
    def _environment_integer(names: tuple[str, ...], default: int) -> int:
        """Return the first valid integer from the given environment variables."""
        for name in names:
            value = os.getenv(name)
            if value is not None:
                try:
                    return int(value)
                except ValueError:
                    continue
        return default


@dataclass(frozen=True)
class DistributedOutcome:
    """Collective success or failure information for one distributed operation."""

    errors: tuple[str | None, ...]

    @property
    def succeeded(self) -> bool:
        """Return whether every rank completed the operation successfully."""
        return all(error is None for error in self.errors)

    @property
    def failed_ranks(self) -> tuple[int, ...]:
        """Return ranks that reported an error."""
        return tuple(rank for rank, error in enumerate(self.errors) if error is not None)

    @property
    def out_of_space(self) -> bool:
        """Return whether any rank reported an out-of-space failure."""
        return _OUT_OF_SPACE_ERROR in self.errors

    def raise_if_failed(
        self,
        operation: str,
        local_error: Exception | None = None,
        error_type: type[Exception] = RuntimeError,
    ) -> None:
        """Raise the local error or a rank-aware error when another rank failed."""
        if self.succeeded:
            return
        if self.out_of_space:
            if isinstance(local_error, OSError) and local_error.errno == errno.ENOSPC:
                raise local_error
            raise OSError(errno.ENOSPC, f"{operation} ran out of space on another rank")
        if local_error is not None:
            raise local_error
        raise error_type(f"{operation} failed on ranks {list(self.failed_ranks)}")


def distributed_context() -> DistributedContext:
    """Return the current non-owning distributed context."""
    return DistributedContext._detect()


class DistributedCoordinator:
    """Coordinate operations across ranks or execute them locally in one process.

    Multi-process execution uses the application-owned default process group and requires it to be initialized by the
    application. In single-process execution, collectives degrade to local behavior: barriers are no-ops, values and
    errors are returned as one-element collections, and local failures are raised directly. AITune never initializes
    or destroys a process group.
    """

    def __init__(self, context: DistributedContext | None = None):
        """Optionally bind to an explicit context; otherwise detect it lazily."""
        self._context = self._validate_context(context) if context is not None else None

    def barrier(self) -> None:
        """Synchronize ranks when running in multiple processes."""
        context = self._current_context()
        if context.is_multi_process:
            dist.barrier()

    def outcome(self, local_error: Exception | None) -> DistributedOutcome:
        """Gather one local exception into a single collective operation outcome."""
        errors = self.collect(_error_token(local_error))
        return DistributedOutcome(tuple(errors))

    def collect_results(
        self,
        value: Any,
        local_error: Exception | None,
    ) -> tuple[DistributedOutcome, list[Any]]:
        """Collect one result and its error status in a single collective."""
        error = _error_token(local_error)
        gathered = self.collect((error, value))
        errors, values = zip(*gathered, strict=True)
        return DistributedOutcome(tuple(errors)), list(values)

    @contextmanager
    def raise_if_any_rank_fails(
        self,
        operation: str,
        error_type: type[Exception] = RuntimeError,
    ) -> Iterator[None]:
        """Raise collectively after a rank fails within the managed operation."""
        local_error: Exception | None = None
        try:
            yield
        except Exception as error:
            local_error = error

        self.outcome(local_error).raise_if_failed(operation, local_error, error_type=error_type)

    def collect(self, value: Any) -> list[Any]:
        """Collect a small control-plane value from every rank."""
        context = self._current_context()
        if not context.is_multi_process:
            return [value]
        values: list[Any] = [None] * context.world_size
        dist.all_gather_object(values, value)
        return values

    def decide_by_rank0(self, rank_zero_decision: bool) -> bool:
        """Broadcast rank zero's control-flow decision to every rank."""
        context = self._current_context()
        if not context.is_multi_process:
            return rank_zero_decision
        decision = [rank_zero_decision if context.rank == 0 else None]
        dist.broadcast_object_list(decision, src=0)
        return bool(decision[0])

    def verify_equal(self, value: Any, description: str) -> None:
        """Raise when ranks do not report the same control-plane value."""
        values = self.collect(value)
        if any(candidate != values[0] for candidate in values[1:]):
            details = ", ".join(f"rank {rank}: {candidate!r}" for rank, candidate in enumerate(values))
            raise RuntimeError(f"Distributed {description} differs across ranks: {details}")

    def _current_context(self) -> DistributedContext:
        """Return the bound context or the current application-owned context."""
        return self._context or self._validate_context(distributed_context())

    @staticmethod
    def _validate_context(context: DistributedContext) -> DistributedContext:
        """Require an application-initialized process group for multi-process tuning."""
        if context.is_multi_process and not context.torch_distributed_initialized:
            raise RuntimeError(
                "Multi-process tuning requires an application-initialized torch.distributed process group. "
                "AITune does not initialize one."
            )
        return context


def resolve_tuning_device(
    requested_device: str | torch.device | None,
    module: nn.Module | None = None,
    context: DistributedContext | None = None,
) -> torch.device:
    """Resolve a tuning device without moving a module or changing CUDA state.

    Resolution order:
        1. Existing device of a distributed module.
        2. Explicitly requested device.
        3. Existing CUDA device of an ordinary module.
        4. Rank-local CUDA device.

    Raises:
        RuntimeError: If no device is requested or inferred and CUDA is unavailable.
    """
    context = context or distributed_context()
    module_device = get_module_device(module) if module is not None else None

    # Distributed modules retain their application-managed placement.
    if module is not None and is_distributed_module(module) and module_device is not None:
        return module_device

    # An explicit request takes precedence for ordinary modules.
    if requested_device is not None:
        device = torch.device(requested_device)
        if device.type == "cuda" and device.index is None:
            # Bind an unindexed CUDA request without changing the active device.
            index = torch.cuda.current_device() if torch.cuda.is_available() else context.local_rank
            return torch.device("cuda", index)
        return device

    # Otherwise prefer the module's CUDA placement, then the rank-local device.
    if module_device is not None and module_device.type == "cuda":
        return module_device
    if torch.cuda.is_available():
        return torch.device("cuda", context.local_rank)
    raise RuntimeError(
        "Cannot resolve an implicit tuning device because CUDA is unavailable. Specify a device explicitly."
    )


def distributed_cache_dir(path: str | Path, context: DistributedContext | None = None) -> Path:
    """Return a reusable topology- and rank-isolated cache path."""
    path = Path(path)
    context = context or distributed_context()
    if not context.is_multi_process:
        return path
    return path / f"rank-{context.rank}-of-{context.world_size}"


def distributed_output_path(path: str | Path, context: DistributedContext | None = None) -> Path:
    """Return a rank-specific output filename for multi-process execution."""
    path = Path(path)
    context = context or distributed_context()
    if not context.is_multi_process:
        return path
    return path.with_name(f"{path.stem}.rank-{context.rank}-of-{context.world_size}{path.suffix}")


coordinator = DistributedCoordinator()


def _error_token(error: Exception | None) -> str | None:
    """Return a bounded error description suitable for a collective."""
    if error is None:
        return None
    if isinstance(error, OSError) and error.errno == errno.ENOSPC:
        return _OUT_OF_SPACE_ERROR
    return error.__class__.__name__
