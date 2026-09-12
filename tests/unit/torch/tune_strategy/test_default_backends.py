# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module-dependent strategy defaults."""

import pytest
import torch

from aitune.torch.backend import (
    TensorRTBackend,
    TorchEagerBackend,
    TorchInductorAotBackend,
    TorchInductorJitBackend,
    TorchTensorRTAotBackend,
)
from aitune.torch.jit.config import Config
from aitune.torch.module.forward_signature import ForwardSignature
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.module.sample_store import SampleStore
from aitune.torch.module.wrapper_module import Module
from aitune.torch.tune_strategy import (
    FirstWinsStrategy,
    LatencyBudgetStrategy,
    MaxThroughputStrategy,
    MinLatencyStrategy,
)


class DistributedModule(torch.nn.Identity):
    __module__ = "torch.distributed.test"


@pytest.fixture(autouse=True)
def backend_availability(mocker):
    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.assert_cuda_is_available")
    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.assert_torch_tensorrt")


@pytest.fixture(params=[FirstWinsStrategy, MaxThroughputStrategy, MinLatencyStrategy, LatencyBudgetStrategy])
def strategy_factory(request):
    strategy_type = request.param
    if strategy_type is LatencyBudgetStrategy:
        return lambda **kwargs: strategy_type(latency_budget_ms=10, **kwargs)
    return strategy_type


@pytest.fixture
def dry_run(tmp_path):
    inputs = torch.ones(1, 2)
    samples = SampleStore.from_samples([((inputs,), {})], tmp_path, "samples")
    graph_spec = GraphSpec(
        name="identity",
        input_spec=SampleMetadata.from_inputs({"input": inputs}),
        output_spec=SampleMetadata.from_outputs(inputs),
        forward_signature=ForwardSignature.from_callable(torch.nn.Identity().forward),
    )

    def run(strategy, module):
        strategy.tune_dry_run(module, "identity", graph_spec, samples, torch.device("cpu"), tmp_path)
        return strategy.to_json_dict()["backends"]

    return run


def test_distributed_dry_run_excludes_native_tensorrt(strategy_factory, mocker, dry_run):
    strategy = strategy_factory()
    defaults_module = "first_wins_strategy" if isinstance(strategy, FirstWinsStrategy) else "profiling_tune_strategy"
    native_tensorrt = mocker.patch(
        f"aitune.torch.tune_strategy.{defaults_module}.TensorRTBackend",
        side_effect=AssertionError,
    )
    descriptions = dry_run(strategy, DistributedModule())

    native_tensorrt.assert_not_called()
    expected = (
        [TorchInductorAotBackend, TorchInductorJitBackend]
        if isinstance(strategy, FirstWinsStrategy)
        else [
            TorchInductorAotBackend,
            TorchTensorRTAotBackend,
            TorchInductorJitBackend,
        ]
    )
    assert [type(backend) for backend in strategy._backends] == expected
    assert descriptions == [backend.describe() for backend in strategy._backends]


@pytest.mark.parametrize("empty", [False, True])
def test_explicit_candidates_are_preserved(strategy_factory, empty, dry_run):
    backends = [] if empty else [TorchEagerBackend()]
    strategy = strategy_factory(backends=backends)

    descriptions = dry_run(strategy, DistributedModule())

    assert descriptions == [backend.describe() for backend in backends]


def test_reused_strategy_restores_ordinary_defaults(strategy_factory, dry_run):
    strategy = strategy_factory()
    ordinary = dry_run(strategy, torch.nn.Identity())
    distributed = dry_run(strategy, DistributedModule())

    assert distributed != ordinary
    assert dry_run(strategy, torch.nn.Identity()) == ordinary


def test_aot_and_jit_default_to_max_throughput_with_separate_instances(dry_run):
    first = Module(torch.nn.Identity(), "first")
    second = Module(torch.nn.Identity(), "second")
    jit = Config().resolve_strategy()

    assert isinstance(first._self_strategy, MaxThroughputStrategy)
    assert isinstance(second._self_strategy, MaxThroughputStrategy)
    assert first._self_strategy is not second._self_strategy
    assert isinstance(jit, MaxThroughputStrategy)
    assert jit is not first._self_strategy
    dry_run(jit, DistributedModule())
    assert [type(backend) for backend in jit._backends] == [TorchInductorAotBackend, TorchInductorJitBackend]


@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("factory", [FirstWinsStrategy, FirstWinsStrategy.for_aot, FirstWinsStrategy.for_jit])
def test_first_wins_owns_its_fallback_order(distributed, factory, dry_run):
    strategy = factory()
    dry_run(strategy, DistributedModule() if distributed else torch.nn.Identity())

    if distributed:
        assert [type(backend) for backend in strategy._backends] == [TorchInductorAotBackend, TorchInductorJitBackend]
    else:
        assert [type(backend) for backend in strategy._backends] == [
            TensorRTBackend,
            TensorRTBackend,
            TorchInductorJitBackend,
        ]
        assert [backend._config.use_dynamo for backend in strategy._backends[:2]] == [True, False]


def test_jit_strategy_clone_preserves_mode_across_topology_changes(dry_run):
    original = MaxThroughputStrategy.for_jit()
    ordinary = dry_run(original, torch.nn.Identity())
    strategy = original.clone()
    assert isinstance(strategy, MaxThroughputStrategy)

    distributed = dry_run(strategy, DistributedModule())
    assert distributed == [TorchInductorAotBackend().describe(), TorchInductorJitBackend().describe()]
    assert original.to_json_dict()["backends"] == ordinary
    assert dry_run(strategy, torch.nn.Identity()) == ordinary

    strategy.enable_performance_validation(False)
    assert strategy.to_json_dict() != original.to_json_dict()


def test_jit_defaults_are_available_during_construction():
    class ReportingStrategy(MaxThroughputStrategy):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.initial_configuration = self.to_json_dict()

    expected = MaxThroughputStrategy.for_jit().to_json_dict()
    strategy = ReportingStrategy.for_jit()

    assert strategy.initial_configuration == expected
    assert strategy.to_json_dict() == expected


@pytest.mark.parametrize("distributed", [False, True])
def test_jit_default_candidate_configuration(distributed, dry_run):
    strategy = Config().resolve_strategy()
    assert isinstance(strategy, MaxThroughputStrategy)
    assert strategy._enable_find_max_batch_size is False
    dry_run(strategy, DistributedModule() if distributed else torch.nn.Identity())
    backends = strategy._backends

    if distributed:
        assert [type(backend) for backend in backends] == [TorchInductorAotBackend, TorchInductorJitBackend]
        assert backends[0]._config.inductor_configs is None
    else:
        assert [type(backend) for backend in backends] == [TensorRTBackend, TensorRTBackend, TorchInductorJitBackend]
        assert [backend._config.use_dynamo for backend in backends[:2]] == [True, False]
    assert backends[-1]._config.mode is None


@pytest.mark.parametrize("distributed", [False, True])
def test_aot_default_candidate_configuration(distributed, dry_run):
    wrapped = Module(DistributedModule() if distributed else torch.nn.Identity(), "model")
    strategy = wrapped._self_strategy
    assert isinstance(strategy, MaxThroughputStrategy)
    dry_run(strategy, wrapped.__wrapped__)
    backends = strategy._backends

    expected = [
        TorchInductorAotBackend,
        TorchTensorRTAotBackend,
        TorchInductorJitBackend,
    ]
    assert [type(backend) for backend in backends] == (expected if distributed else [TensorRTBackend] * 2 + expected)
    if not distributed:
        assert [backend._config.use_dynamo for backend in backends[:2]] == [True, False]
    aot, torch_trt, jit = backends[-3:]
    assert aot._config.inductor_configs is None
    assert torch_trt._compile_settings()["use_distributed_mode_trace"] is distributed
    assert jit._config.mode is None
    assert len({backend.key() for backend in backends}) == len(backends)


@pytest.mark.parametrize(
    "strategy_type", [FirstWinsStrategy, MaxThroughputStrategy, MinLatencyStrategy, LatencyBudgetStrategy]
)
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("workflow", ["aot", "jit"])
def test_workflow_factory_preserves_explicit_configuration(strategy_type, empty, workflow, dry_run):
    backends = [] if empty else [TorchEagerBackend()]
    kwargs = {"latency_budget_ms": 10} if strategy_type is LatencyBudgetStrategy else {}
    factory = {"aot": strategy_type.for_aot, "jit": strategy_type.for_jit}[workflow]
    strategy = factory(backends=backends, **kwargs)

    assert type(strategy) is strategy_type
    assert strategy._enable_find_max_batch_size is (workflow == "aot" and strategy_type is not MinLatencyStrategy)
    expected = [backend.describe() for backend in backends]
    assert dry_run(strategy, torch.nn.Identity()) == expected
    assert dry_run(strategy, DistributedModule()) == expected


@pytest.mark.parametrize("strategy_type", [MinLatencyStrategy, LatencyBudgetStrategy])
@pytest.mark.parametrize("distributed", [False, True])
def test_latency_strategies_compare_the_same_candidates_in_both_workflows(strategy_type, distributed, dry_run):
    kwargs = {"latency_budget_ms": 10} if strategy_type is LatencyBudgetStrategy else {}
    aot = strategy_type.for_aot(**kwargs)
    jit = strategy_type.for_jit(**kwargs)
    module = DistributedModule() if distributed else torch.nn.Identity()
    assert dry_run(jit, module) == dry_run(aot, module)

    jit.enable_performance_validation(False)
    assert jit.to_json_dict() != aot.to_json_dict()
