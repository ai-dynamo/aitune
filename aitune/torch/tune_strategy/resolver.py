# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create tune strategies whose backends are resolved for each module."""

from collections.abc import Sequence
from typing import Literal, Self

import torch.nn as nn
from pydantic import BaseModel, ConfigDict, PositiveFloat, field_validator, model_validator, validate_call

from aitune.torch.backend import (
    Backend,
    ONNXRuntimeBackend,
    TensorRTBackend,
    TensorRTBackendConfig,
    TorchInductorAotBackend,
    TorchInductorJitBackend,
)
from aitune.torch.backend.backend import BuildMode, ExecutionMode, ModuleFormat
from aitune.torch.module.onnx_module import OnnxModule
from aitune.torch.tune_strategy.latency_budget_strategy import LatencyBudgetStrategy
from aitune.torch.tune_strategy.max_throughput_strategy import MaxThroughputStrategy
from aitune.torch.tune_strategy.min_latency_strategy import MinLatencyStrategy
from aitune.torch.tune_strategy.mixin.find_max_batch_size_mixin import FindMaxBatchSizeMixin
from aitune.torch.tune_strategy.tune_strategy import TuneStrategy
from aitune.torch.utils.module import is_distributed_module

Relation = Literal["at_most"]
Objective = Literal["throughput", "latency"]
Compilation = Literal["aot", "jit", "any"]
Unit = Literal["ms"]

_OBJECTIVES = frozenset({"throughput", "latency"})
_COMPILATION_MODES = frozenset({"aot", "jit", "any"})


class Constraint(BaseModel):
    """Describe one requirement that a tuning candidate must satisfy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: Objective
    relation: Relation
    value: PositiveFloat
    unit: Unit = "ms"

    @model_validator(mode="after")
    def _metric_and_unit_validation(self) -> Self:
        """Only 'latency' metric with 'ms' unit is supported for now."""
        if not (self.metric == "latency" and self.unit == "ms"):
            raise ValueError("Only 'latency' metric with 'ms' unit is supported for now.")
        return self

    @classmethod
    def max_latency_ms(cls, milliseconds: float) -> Self:
        """Require latency to be no greater than ``milliseconds``."""
        return cls(metric="latency", relation="at_most", value=milliseconds, unit="ms")


class DynamicTuneStrategy(BaseModel):
    """Describes rules that are used to select and configure an optimal tuning strategy for a given module at tune time.

    This class holds no backend instances or backend-specific state. Instead, it contains a declarative description of
    the desired optimization *objective* (such as "throughput" or "latency") and a set of *constraints*, all of which apply together.

    In this framework:
      - The *objective* defines what we try to maximize or minimize (e.g., maximize throughput).
      - The *constraint(s)* express requirements that must be met by any configuration the strategy chooses (such as "latency at_most N ms").
        For now, only a single latency constraint is supported (e.g., "latency at_most N ms").
      - Typically, this means we optimize for maximum throughput, but only among strategies/configurations that meet the given latency constraint.

    This class is *dynamic*: it does not directly reference any backends.
    When `materialize()` is called with a particular module, it creates a concrete `TuneStrategy` with appropriate backend objects
    and automatically selects the right strategy class (e.g., `LatencyBudgetStrategy` for a throughput-optimization subject to a latency constraint).

    The same DynamicTuneStrategy instance can be used for multiple modules, and a fresh concrete strategy is produced for each module.

    Example:
        # Optimize for max throughput, but only if latency ≤ 40 ms for the current module
        dts = DynamicTuneStrategy(
            objective="throughput",
            compilation="any",
            constraints=[Constraint.max_latency_ms(40)]
        )
        strategy = dts.materialize(some_module)
    """

    objective: Objective
    compilation: Compilation
    constraints: Sequence[Constraint]
    _find_max_batch_size: bool | None = None

    @field_validator("constraints", mode="before")
    @classmethod
    def _validate_constraints(cls, v):
        return tuple(v)

    def enable_find_max_batch_size(self, enable: bool = True) -> Self:
        """Configure maximum-batch-size discovery on the resolved strategy."""
        self._find_max_batch_size = enable
        return self

    def materialize(self, module: nn.Module) -> TuneStrategy:
        """Create a concrete strategy with fresh backends for ``module``."""
        backends = _resolve_backends(module, compilation=self.compilation)

        # Only one constraint with metric='latency' and relation='at_most' is supported.
        latency_limit = next(iter(self.constraints), None)

        if self.objective == "latency":
            strategy: TuneStrategy = MinLatencyStrategy(backends=backends)
        elif latency_limit is not None:
            strategy = LatencyBudgetStrategy(
                backends=backends,
                latency_budget_ms=latency_limit.value,
            )
        else:
            strategy = MaxThroughputStrategy(backends=backends)

        if self._find_max_batch_size is not None and isinstance(strategy, FindMaxBatchSizeMixin):
            strategy.enable_find_max_batch_size(self._find_max_batch_size)

        return strategy

    def clone(self) -> Self:
        """Return an independent copy of this dynamic configuration."""
        return self.model_copy(deep=True)

    def to_json_dict(self) -> dict:
        """Return the unresolved configuration for reporting."""
        dump = self.model_dump(mode="json", exclude={"_find_max_batch_size"})
        dump["backends"] = "resolved for each module"
        return dump


StrategyOption = TuneStrategy | DynamicTuneStrategy


@validate_call
def resolve_strategy(
    *,
    objective: Objective = "throughput",
    compilation: Compilation = "any",
    constraints: Sequence[Constraint] = (),
) -> DynamicTuneStrategy:
    """Declare a strategy whose backends will be resolved for each tuned module.

    Args:
        objective: Performance metric to optimize: throughput or latency.
        compilation: Allow ahead-of-time, just-in-time, or any supported backend build mode.
        constraints: Limits that candidates must satisfy while optimizing the objective.

    Returns:
        A dynamic strategy configuration suitable for AOT and JIT tuning.
    """
    seen_constraints: set[tuple[str, str]] = set()

    # Validate all constraints: only one per (metric, relation) pair, must be supported,
    # Latency constraint with throughput objective.
    for constraint in constraints:
        constraint_key = (constraint.metric, constraint.relation)
        if constraint_key in seen_constraints:
            raise ValueError(
                f"Only one constraint with metric={constraint.metric!r} and relation={constraint.relation!r} "
                "may be specified."
            )
        if constraint_key != ("latency", "at_most") or constraint.unit != "ms":
            raise ValueError(
                f"Unsupported constraint: metric={constraint.metric!r}, relation={constraint.relation!r}, "
                f"unit={constraint.unit!r}."
            )
        if objective != "throughput":
            raise ValueError("The maximum latency constraint is only valid with objective='throughput'.")

        seen_constraints.add(constraint_key)

    return DynamicTuneStrategy(objective=objective, compilation=compilation, constraints=constraints)


def materialize_strategy(strategy: StrategyOption, module: nn.Module) -> TuneStrategy:
    """Resolve a dynamic strategy for ``module`` or preserve an explicit strategy."""
    if isinstance(strategy, DynamicTuneStrategy):
        return strategy.materialize(module)
    return strategy


def _resolve_backends(module: nn.Module, *, compilation: Compilation) -> list[Backend]:
    """Create default candidates compatible with the module and requested compilation."""
    module_format = ModuleFormat.ONNX if isinstance(module, OnnxModule) else ModuleFormat.TORCH
    if module_format is ModuleFormat.ONNX:
        backends: list[Backend] = [
            TensorRTBackend(),
            ONNXRuntimeBackend(),
        ]
    else:
        backends = [
            TensorRTBackend(),
            TensorRTBackend(config=TensorRTBackendConfig(use_dynamo=False)),
            TorchInductorAotBackend(),
            TorchInductorJitBackend(),
        ]

    execution_mode = ExecutionMode.MULTI_GPU if is_distributed_module(module) else ExecutionMode.SINGLE_GPU
    backends = [backend for backend in backends if execution_mode in backend._execution_modes]

    build_mode = {
        "aot": BuildMode.AHEAD_OF_TIME,
        "jit": BuildMode.JUST_IN_TIME,
    }.get(compilation)
    if build_mode is not None:
        backends = [backend for backend in backends if backend.build_mode is build_mode]

    if not backends:
        raise RuntimeError(
            f"No default backends support module {module.__class__.__qualname__!r} with compilation={compilation!r}."
        )
    return backends
