# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TorchDynamo configuration for explicitly registered JIT targets."""

from threading import RLock
from typing import Any

from torch._dynamo import config as dynamo_config

from aitune.torch.tune_strategy.tune_strategy import TuneStrategy

_LIMIT_NAMES = (
    "cache_size_limit",
    "recompile_limit",
    "accumulated_cache_size_limit",
    "accumulated_recompile_limit",
)


def _backend_candidate_count(strategy: TuneStrategy) -> int:
    """Count backend candidates from a strategy already configured for its module."""
    strategy_config: dict[str, Any] = strategy.to_json_dict()
    backends = strategy_config.get("backends")
    if isinstance(backends, list):
        return max(1, len(backends))
    return 1


class _SessionDynamoRecompileCapacity:
    """Own Dynamo limit reservations until selected JIT routes have run."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._config: Any | None = None
        self._original_limits: dict[str, int] = {}
        self._assigned_limits: dict[str, int] = {}
        self._owners: set[int] = set()

    def reserve(self, owner: int, required_limit: int) -> None:
        """Raise supported limits without shortening an existing reservation."""
        with self._lock:
            if self._config is not None and self._config is not dynamo_config:
                self._restore_locked()
            if self._config is None:
                self._config = dynamo_config
                self._original_limits = {
                    name: current
                    for name in _LIMIT_NAMES
                    if isinstance((current := getattr(self._config, name, None)), int)
                }
            self._owners.add(owner)

            for name in self._original_limits:
                current = getattr(self._config, name, None)
                if not isinstance(current, int):
                    continue

                assigned_limit = self._assigned_limits.get(name)
                if (assigned_limit is None and current != self._original_limits[name]) or (
                    assigned_limit is not None and current != assigned_limit
                ):
                    # Other code changed the limit while the JIT session was active.
                    # Preserve that value as the new restoration baseline.
                    self._original_limits[name] = current
                    self._assigned_limits.pop(name, None)

                if current < required_limit:
                    setattr(self._config, name, required_limit)
                    self._assigned_limits[name] = required_limit

    def release(self, owner: int) -> None:
        """Release one target after its required selected-inference routes."""
        with self._lock:
            self._owners.discard(owner)
            if not self._owners:
                self._restore_locked()

    def restore(self) -> None:
        """Restore limits changed by this reservation and clear session state."""
        with self._lock:
            self._restore_locked()

    def _restore_locked(self) -> None:
        """Restore limits still owned by this session, then clear its reservations."""
        if self._config is not None:
            for name, assigned_limit in self._assigned_limits.items():
                if getattr(self._config, name, None) == assigned_limit:
                    setattr(self._config, name, self._original_limits[name])
        self._config = None
        self._original_limits.clear()
        self._assigned_limits.clear()
        self._owners.clear()


_SESSION_CAPACITY = _SessionDynamoRecompileCapacity()


def reserve_dynamo_recompile_capacity(
    owner: int,
    route_count: int,
    strategy: TuneStrategy,
    *,
    detect_graph_breaks: bool = False,
) -> None:
    """Reserve Dynamo capacity through a target's selected JIT inference routes.

    Many instances of the same module class share one Python ``forward`` code object. Explicit registration makes
    those instances independent compilation routes, which can exceed TorchDynamo's default per-frame recompile limit.
    The selected backend can need one additional route after tuning patches the module's device attribute, and graph
    break detection consumes another route when enabled. Capacity therefore includes backend candidates, detection, and
    the deployed inference variant. The caller keeps the reservation active until
    every selected JIT route runs successfully once because TorchDynamo may compile
    those variants lazily. It is also released by :func:`jit_reset`.

    Args:
        owner: Stable identifier of the explicit JIT target using the reservation.
        route_count: Number of explicit module/graph routes currently registered.
        strategy: Strategy whose backend candidates may compile each route.
        detect_graph_breaks: Whether each route may first be compiled for graph-break detection.
    """
    compile_variants = _backend_candidate_count(strategy) + 1 + int(detect_graph_breaks)
    required_limit = max(1, route_count) * compile_variants
    _SESSION_CAPACITY.reserve(owner, required_limit)


def release_dynamo_recompile_capacity(owner: int) -> None:
    """Release one explicit target's selected-inference capacity."""
    _SESSION_CAPACITY.release(owner)


def restore_dynamo_recompile_capacity() -> None:
    """Release the explicit JIT session's Dynamo capacity reservation."""
    _SESSION_CAPACITY.restore()
