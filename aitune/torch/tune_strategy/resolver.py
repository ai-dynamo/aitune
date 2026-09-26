# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create tune strategies whose backends are resolved for each module."""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch.nn as nn

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

Objective = Literal["throughput", "latency"]
Compilation = Literal["aot", "jit", "any"]

_OBJECTIVES = frozenset({"throughput", "latency"})
_COMPILATION_MODES = frozenset({"aot", "jit", "any"})


@dataclass(frozen=True)
class Constraint:
    """Describe one requirement that a tuning candidate must satisfy."""

    metric: str
    relation: str
    value: float
    unit: str

    @classmethod
    def max_latency_ms(cls, milliseconds: float) -> "Constraint":
        """Require latency to be no greater than ``milliseconds``."""
        if milliseconds <= 0:
            raise ValueError(f"Maximum latency must be greater than zero, got {milliseconds}.")
        return cls(metric="latency", relation="at_most", value=milliseconds, unit="ms")


class DynamicTuneStrategy:
    """Describe a strategy whose concrete backends depend on the tuned module.

    Instances contain no backend state and may be shared between wrapped modules after
    configuration. A fresh concrete ``TuneStrategy`` is created immediately before each
    module is tuned.
    """

    def __init__(
        self,
        *,
        objective: Objective,
        compilation: Compilation,
        constraints: Sequence[Constraint],
    ) -> None:
        """Store user intent without creating module-specific backends."""
        self.objective = objective
        self.compilation = compilation
        self.constraints = tuple(constraints)
        self._find_max_batch_size: bool | None = None

    def enable_find_max_batch_size(self, enable: bool = True) -> "DynamicTuneStrategy":
        """Configure maximum-batch-size discovery on the resolved strategy."""
        self._find_max_batch_size = enable
        return self

    def materialize(self, module: nn.Module) -> TuneStrategy:
        """Create a concrete strategy with fresh backends for ``module``."""
        backends = _resolve_backends(module, compilation=self.compilation)
        latency_limit = next(
            (
                constraint
                for constraint in self.constraints
                if constraint.metric == "latency" and constraint.relation == "at_most"
            ),
            None,
        )
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

    def clone(self) -> "DynamicTuneStrategy":
        """Return an independent copy of this dynamic configuration."""
        return deepcopy(self)

    def to_json_dict(self) -> dict:
        """Return the unresolved configuration for reporting."""
        return {
            "objective": self.objective,
            "compilation": self.compilation,
            "constraints": [
                {
                    "metric": constraint.metric,
                    "relation": constraint.relation,
                    "value": constraint.value,
                    "unit": constraint.unit,
                }
                for constraint in self.constraints
            ],
            "backends": "resolved for each module",
        }


StrategyInput = TuneStrategy | DynamicTuneStrategy


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
    _validate_choice("objective", objective, _OBJECTIVES)
    _validate_choice("compilation", compilation, _COMPILATION_MODES)
    constraints = tuple(constraints)
    seen_constraints: set[tuple[str, str]] = set()
    for constraint in constraints:
        if not isinstance(constraint, Constraint):
            raise TypeError(f"Unsupported constraint type: {type(constraint).__qualname__}.")
        constraint_key = (constraint.metric, constraint.relation)
        if constraint_key in seen_constraints:
            raise ValueError(
                f"Only one constraint with metric={constraint.metric!r} and relation={constraint.relation!r} "
                "may be specified."
            )
        seen_constraints.add(constraint_key)
        if constraint_key != ("latency", "at_most") or constraint.unit != "ms":
            raise ValueError(
                f"Unsupported constraint: metric={constraint.metric!r}, relation={constraint.relation!r}, "
                f"unit={constraint.unit!r}."
            )
        if objective != "throughput":
            raise ValueError("The maximum latency constraint is only valid with objective='throughput'.")

    return DynamicTuneStrategy(
        objective=objective,
        compilation=compilation,
        constraints=constraints,
    )


def materialize_strategy(strategy: StrategyInput, module: nn.Module) -> TuneStrategy:
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

    # Source-format compatibility is the first capability filter. In particular,
    # an OnnxModule must never reach a backend that expects a torch.nn.Module graph.
    backends = [backend for backend in backends if module_format in backend._supported_modules]

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


def _validate_choice(name: str, value: str, choices: frozenset[str]) -> None:
    """Validate one public string option with a useful error."""
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"Unknown {name} {value!r}; expected one of: {expected}.")
