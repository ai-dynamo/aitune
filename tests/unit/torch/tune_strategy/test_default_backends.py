# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for dynamic strategy and backend resolution."""

from unittest.mock import Mock

import pytest
import torch

from aitune.torch.backend import (
    ModuleFormat,
    ONNXRuntimeBackend,
    TensorRTBackend,
    TorchInductorAotBackend,
    TorchInductorJitBackend,
)
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.onnx_module import OnnxModule
from aitune.torch.module.wrapper_module import Module
from aitune.torch.tune_strategy import (
    Constraint,
    LatencyBudgetStrategy,
    MaxThroughputStrategy,
    MinLatencyStrategy,
    resolve_strategy,
)
from aitune.torch.tune_strategy.resolver import materialize_strategy


class DistributedModule(torch.nn.Identity):
    __module__ = "torch.distributed.test"


def test_resolve_strategy_captures_dynamic_request():
    strategy = resolve_strategy(objective="throughput", compilation="mixed")

    assert strategy.to_json_dict() == {
        "objective": "throughput",
        "compilation": "mixed",
        "constraints": [],
        "backends": "resolved for each module",
    }


@pytest.mark.parametrize(
    "compilation,expected",
    [
        ("aot", [TensorRTBackend, TensorRTBackend, TorchInductorAotBackend]),
        ("jit", [TorchInductorJitBackend]),
        ("mixed", [TensorRTBackend, TensorRTBackend, TorchInductorAotBackend, TorchInductorJitBackend]),
    ],
)
def test_dynamic_strategy_uses_compilation_mode(compilation, expected):
    strategy = materialize_strategy(resolve_strategy(compilation=compilation), torch.nn.Identity())

    assert isinstance(strategy, MaxThroughputStrategy)
    assert [type(backend) for backend in strategy._backends] == expected


def test_dynamic_strategy_uses_distributed_candidates():
    strategy = materialize_strategy(resolve_strategy(), DistributedModule())

    assert isinstance(strategy, MaxThroughputStrategy)
    assert [type(backend) for backend in strategy._backends] == [TorchInductorAotBackend, TorchInductorJitBackend]


def test_dynamic_strategy_uses_only_onnx_compatible_candidates(tmp_path):
    module = OnnxModule(tmp_path / "model.onnx")

    strategy = materialize_strategy(resolve_strategy(), module)

    assert isinstance(strategy, MaxThroughputStrategy)
    assert [type(backend) for backend in strategy._backends] == [
        TensorRTBackend,
        ONNXRuntimeBackend,
    ]
    assert all(ModuleFormat.ONNX in backend._supported_modules for backend in strategy._backends)


def test_dynamic_strategy_rejects_jit_compilation_for_onnx_module(tmp_path):
    module = OnnxModule(tmp_path / "model.onnx")

    with pytest.raises(RuntimeError, match="No default backends support module 'OnnxModule'.*compilation='jit'"):
        materialize_strategy(resolve_strategy(compilation="jit"), module)


def test_dynamic_strategy_returns_fresh_backends():
    request = resolve_strategy()
    first = materialize_strategy(request, torch.nn.Identity())
    second = materialize_strategy(request, torch.nn.Identity())

    assert isinstance(first, MaxThroughputStrategy)
    assert isinstance(second, MaxThroughputStrategy)
    assert first is not second
    assert all(left is not right for left, right in zip(first._backends, second._backends, strict=True))


def test_aot_wrapper_materializes_strategy_for_each_module():
    request = resolve_strategy()
    ordinary = Module(torch.nn.Identity(), "ordinary", strategy=request)
    distributed = Module(DistributedModule(), "distributed", strategy=request)
    graph_spec = Mock(spec=GraphSpec)

    ordinary_strategy = ordinary._get_strategies_for_graph_specs(None, [graph_spec], dry_run=False)[0]
    distributed_strategy = distributed._get_strategies_for_graph_specs(None, [graph_spec], dry_run=False)[0]

    assert isinstance(ordinary_strategy, MaxThroughputStrategy)
    assert isinstance(distributed_strategy, MaxThroughputStrategy)
    assert [type(backend) for backend in ordinary_strategy._backends] == [
        TensorRTBackend,
        TensorRTBackend,
        TorchInductorAotBackend,
        TorchInductorJitBackend,
    ]
    assert [type(backend) for backend in distributed_strategy._backends] == [
        TorchInductorAotBackend,
        TorchInductorJitBackend,
    ]


def test_dynamic_strategy_applies_find_max_batch_size_configuration():
    request = resolve_strategy().enable_find_max_batch_size(False)

    strategy = materialize_strategy(request, torch.nn.Identity())

    assert isinstance(strategy, MaxThroughputStrategy)
    assert strategy._enable_find_max_batch_size is False


def test_dynamic_strategy_maps_latency_objective():
    strategy = materialize_strategy(resolve_strategy(objective="latency"), torch.nn.Identity())

    assert isinstance(strategy, MinLatencyStrategy)


def test_dynamic_strategy_maps_latency_limit():
    strategy = materialize_strategy(resolve_strategy(constraints=[Constraint.max_latency_ms(20)]), torch.nn.Identity())

    assert isinstance(strategy, LatencyBudgetStrategy)
    assert strategy.latency_budget_ms == 20


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"objective": "unknown"}, "Unknown objective"),
        ({"compilation": "unknown"}, "Unknown compilation"),
        (
            {"objective": "latency", "constraints": [Constraint.max_latency_ms(20)]},
            "only valid with objective='throughput'",
        ),
    ],
)
def test_resolve_strategy_rejects_invalid_options(kwargs, error):
    with pytest.raises(ValueError, match=error):
        resolve_strategy(**kwargs)  # pytype: disable=wrong-arg-types


def test_max_latency_constraint_rejects_non_positive_value():
    with pytest.raises(ValueError, match="Maximum latency must be greater than zero"):
        Constraint.max_latency_ms(0)


def test_resolve_strategy_rejects_duplicate_constraint_types():
    with pytest.raises(ValueError, match="Only one constraint with metric='latency' and relation='at_most'"):
        resolve_strategy(constraints=[Constraint.max_latency_ms(10), Constraint.max_latency_ms(20)])
