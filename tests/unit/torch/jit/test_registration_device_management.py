# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-lifetime placement policy for explicitly registered prepared modules."""

from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch

from aitune.torch import register_for_jit_tuning
from aitune.torch.backend import (
    TorchEagerBackend,
    TorchInductorJitBackend,
    TorchInductorJitBackendConfig,
)
from aitune.torch.backend.backend import BuildMode, DummyBackend, ExecutionMode
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.patcher import Patcher
from aitune.torch.module.onnx_module import OnnxModule
from aitune.torch.tune_strategy import (
    FirstWinsStrategy,
    LatencyBudgetStrategy,
    MaxThroughputStrategy,
    MinLatencyStrategy,
    OneBackendStrategy,
)
from aitune.torch.utils.module import (
    is_externally_managed_module,
    move_module_to_device,
    offload,
)


class _ModuleTree(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.child = torch.nn.Linear(2, 2)

    def forward(self, value):
        return self.child(value)


class _UnsupportedBackend(DummyBackend):
    _execution_modes = frozenset({ExecutionMode.SINGLE_GPU})
    _build_mode = BuildMode.JUST_IN_TIME


class _EagerSubclass(TorchEagerBackend):
    _execution_modes = frozenset({ExecutionMode.SINGLE_GPU})
    _build_mode = BuildMode.JUST_IN_TIME


class _CustomOneBackendStrategy(OneBackendStrategy):
    def _pre_tune(self, *args, **kwargs):
        raise AssertionError("Custom source mutation must never be reached")


class _CustomFirstWinsStrategy(FirstWinsStrategy):
    pass


def _configure(tmp_path):
    jit_config.mode = JITMode.TUNE_DEFERRED
    jit_config.device = torch.device("cpu")
    jit_config.batch_axis_required = False
    jit_config.min_parameters = -1
    jit_config.detect_graph_breaks = False
    jit_config.cache_dir = tmp_path / "jit-cache"
    jit_config.strategy = OneBackendStrategy(TorchEagerBackend()).enable_performance_validation(False)


@pytest.mark.parametrize(
    "strategy",
    [
        OneBackendStrategy(TorchEagerBackend()),
        FirstWinsStrategy([TorchEagerBackend()]),
        MinLatencyStrategy([TorchEagerBackend()]),
        MaxThroughputStrategy([TorchEagerBackend()]),
        LatencyBudgetStrategy(latency_budget_ms=1.0, backends=[TorchEagerBackend()]),
    ],
)
def test_external_registration_accepts_concrete_builtin_strategies(tmp_path, strategy):
    _configure(tmp_path)
    jit_config.strategy = strategy
    source = _ModuleTree()

    registration = register_for_jit_tuning([source], device_management="external")

    assert registration.modules == (source,)
    assert all(is_externally_managed_module(node) for node in source.modules())


@pytest.mark.parametrize(
    "strategy",
    [
        _CustomOneBackendStrategy(TorchEagerBackend()),
        _CustomFirstWinsStrategy([TorchEagerBackend()]),
        FirstWinsStrategy([]),
        MinLatencyStrategy([]),
        MaxThroughputStrategy([]),
        LatencyBudgetStrategy(latency_budget_ms=1.0, backends=[]),
    ],
)
def test_external_registration_rejects_custom_or_empty_strategy_before_installation(tmp_path, strategy):
    _configure(tmp_path)
    jit_config.strategy = strategy
    source = _ModuleTree()

    with pytest.raises(ValueError, match="External device management"):
        register_for_jit_tuning([source], device_management="external")

    assert "forward" not in source.__dict__
    assert not any(is_externally_managed_module(node) for node in source.modules())
    assert Patcher._entry_path is None
    assert Patcher._patched_modules == []

    jit_config.strategy = OneBackendStrategy(TorchEagerBackend())
    registration = register_for_jit_tuning([source], device_management="external")
    assert registration.modules == (source,)


@pytest.mark.parametrize("backend", [TorchEagerBackend(), TorchInductorJitBackend()])
def test_external_registration_rejects_onnx_in_the_complete_batch_before_installation(tmp_path, backend):
    _configure(tmp_path)
    jit_config.strategy = OneBackendStrategy(backend)
    source = _ModuleTree()
    onnx = OnnxModule(tmp_path / "unused.onnx")

    with pytest.raises(RuntimeError, match="does not support onnx modules"):
        register_for_jit_tuning([source, onnx], device_management="external")

    for target in (source, onnx):
        assert "forward" not in target.__dict__
        assert not any(is_externally_managed_module(node) for node in target.modules())
    assert onnx._session is None
    assert onnx._device is None
    assert Patcher._entry_path is None
    assert Patcher._patched_modules == []

    registration = register_for_jit_tuning([source], device_management="external")
    assert registration.modules == (source,)


def test_invalid_device_policy_is_rejected_before_installation(tmp_path):
    _configure(tmp_path)
    source = _ModuleTree()
    with pytest.raises(ValueError, match="device_management"):
        register_for_jit_tuning([source], device_management="application")  # pytype: disable=wrong-arg-types
    assert "forward" not in source.__dict__
    assert not any(is_externally_managed_module(node) for node in source.modules())
    assert Patcher._entry_path is None
    registration = register_for_jit_tuning([source], device_management="external")
    assert registration.modules == (source,)


def test_registration_protects_source_nodes_but_independent_copies_stay_movable(tmp_path, mocker):
    _configure(tmp_path)
    source, unrelated = _ModuleTree(), torch.nn.Linear(2, 2)
    register_for_jit_tuning([source], device_management="external")
    wrapper = Patcher._patched_modules[0]
    wrapper._restore_original_forward()
    try:
        copied = deepcopy(source)
    finally:
        wrapper._proxy_forward()
    moves = [mocker.spy(node, "to") for node in source.modules()]
    cleanup = mocker.patch("aitune.torch.utils.module.cleanup_memory")

    for node in source.modules():
        assert is_externally_managed_module(node)
        move_module_to_device(node, "meta")
        offload(node, "meta")
    assert all(move.call_count == 0 for move in moves)
    cleanup.assert_not_called()
    assert not is_externally_managed_module(copied.child)
    move_module_to_device(copied, "meta")
    move_module_to_device(unrelated, "meta")
    assert copied.child.weight.device.type == unrelated.weight.device.type == "meta"
    assert source.child.weight.device.type == "cpu"


def test_external_eager_tuning_and_serving_keep_source_placement(tmp_path, mocker):
    _configure(tmp_path)
    source = _ModuleTree().eval()
    original_class = type(source)
    original_weight = source.child.weight
    moves = [mocker.spy(node, "to") for node in source.modules()]
    value = torch.ones(1, 2)
    expected = source(value)
    registration = register_for_jit_tuning([source], device_management="external")

    with torch.inference_mode():
        source(value)
        Patcher.enable_tune_deferred()
        source(value)
        torch.testing.assert_close(source(value), expected)

    assert registration.all_tuned
    assert type(source) is original_class
    assert source.child.weight is original_weight
    assert all(move.call_count == 0 for move in moves)
    assert all(is_externally_managed_module(node) for node in source.modules())


@pytest.mark.parametrize("outcome", ["dry-run", "skipped", "eager-fallback"])
def test_external_marks_survive_tuning_skip_and_fallback(tmp_path, mocker, outcome):
    _configure(tmp_path)
    jit_config.dry_run = True
    jit_config.dry_run_failure_probability = 0.0
    dry_run = mocker.patch.object(OneBackendStrategy, "tune_dry_run")
    if outcome == "skipped":
        jit_config.min_parameters = 100
    elif outcome == "eager-fallback":
        dry_run.side_effect = RuntimeError("candidate failed")
    source = _ModuleTree()
    original_class = type(source)
    moves = [mocker.spy(node, "to") for node in source.modules()]
    value = torch.ones(1, 2)
    expected = source(value)
    registration = register_for_jit_tuning([source], device_management="external")
    source(value)
    Patcher.enable_tune_deferred()
    source(value)

    expected_state = {"dry-run": "tuned", "skipped": "skipped", "eager-fallback": "eager"}[outcome]
    assert registration.state_counts == {expected_state: 1}
    torch.testing.assert_close(source(value), expected)
    assert type(source) is original_class
    assert all(is_externally_managed_module(node) for node in source.modules())
    assert all(move.call_count == 0 for move in moves)
    if outcome == "skipped":
        dry_run.assert_not_called()
    else:
        dry_run.assert_called_once()
    with pytest.raises(RuntimeError, match="fresh worker"):
        register_for_jit_tuning([source], device_management="aitune")


@pytest.mark.parametrize(
    "backend",
    [
        _UnsupportedBackend(),
        _EagerSubclass(),
        TorchInductorJitBackend(TorchInductorJitBackendConfig(mode="reduce-overhead")),
        TorchInductorJitBackend(TorchInductorJitBackendConfig(mode="max-autotune")),
        TorchInductorJitBackend(TorchInductorJitBackendConfig(options={"max_autotune": True})),
    ],
)
def test_external_backend_qualification_fails_before_registration_and_direct_build(tmp_path, mocker, backend):
    _configure(tmp_path)
    jit_config.strategy = FirstWinsStrategy([TorchEagerBackend(), backend])
    source = _ModuleTree()
    move = mocker.spy(source, "to")
    with pytest.raises(ValueError, match="(?i)external"):
        register_for_jit_tuning([source], device_management="external")
    assert "forward" not in source.__dict__
    assert not any(is_externally_managed_module(node) for node in source.modules())
    assert Patcher._entry_path is None
    move.assert_not_called()

    jit_config.strategy = OneBackendStrategy(TorchEagerBackend())
    register_for_jit_tuning([source], device_management="external")
    build = mocker.patch.object(backend, "_build", return_value=backend)
    with pytest.raises((ValueError, RuntimeError), match="(?i)external"):
        backend.build(source, Mock(), [], torch.device("cpu"), tmp_path)
    build.assert_not_called()
    move.assert_not_called()


@pytest.mark.parametrize(
    "config",
    [
        TorchInductorJitBackendConfig(),
        TorchInductorJitBackendConfig(mode="default"),
        TorchInductorJitBackendConfig(mode="max-autotune-no-cudagraphs"),
        TorchInductorJitBackendConfig(options={}),
    ],
)
def test_qualified_inductor_configuration_accepts_external_registration(tmp_path, config):
    _configure(tmp_path)
    jit_config.strategy = OneBackendStrategy(TorchInductorJitBackend(config))
    source = _ModuleTree()
    registration = register_for_jit_tuning([source], device_management="external")
    assert registration.modules == (source,)
    assert is_externally_managed_module(source.child)


def test_ordinary_backend_is_not_restricted_by_external_qualification(tmp_path, mocker):
    ordinary = torch.nn.Linear(2, 2)
    backend = _UnsupportedBackend()
    build = mocker.patch.object(backend, "_build", return_value=backend)
    assert backend.build(ordinary, Mock(), [], torch.device("cpu"), tmp_path) is backend
    build.assert_called_once()


def test_backend_rejects_unmarked_ancestor_before_any_build_mutation(tmp_path, mocker):
    _configure(tmp_path)
    source = _ModuleTree()
    ancestor = torch.nn.Sequential(source, torch.nn.Linear(2, 2))
    register_for_jit_tuning([source], device_management="external")
    backend = TorchEagerBackend()
    build = mocker.patch.object(backend, "_build", return_value=backend)
    save = mocker.spy(backend, "_save_config")
    move = mocker.spy(ancestor, "to")
    cleanup = mocker.patch("aitune.torch.backend.backend.release_transient_memory")
    with pytest.raises(RuntimeError, match="unmarked ancestor"):
        backend.build(ancestor, Mock(), [], torch.device("cpu"), tmp_path)
    build.assert_not_called()
    save.assert_not_called()
    move.assert_not_called()
    cleanup.assert_not_called()


def test_default_registration_still_manages_ordinary_source_placement(tmp_path, mocker):
    _configure(tmp_path)
    jit_config.dry_run = True
    jit_config.dry_run_failure_probability = 0.0
    mocker.patch.object(OneBackendStrategy, "tune_dry_run")
    source = _ModuleTree()
    original_class = type(source)
    move = mocker.spy(source, "to")
    registration = register_for_jit_tuning([source])
    value = torch.ones(1, 2)
    source(value)
    Patcher.enable_tune_deferred()
    source(value)

    assert registration.all_tuned
    assert type(source) is not original_class
    move.assert_called_once_with(torch.device("cpu"))
    assert not is_externally_managed_module(source)


def test_external_registration_requires_a_known_candidate_plan(tmp_path):
    _configure(tmp_path)
    jit_config.strategy = Mock()
    source = _ModuleTree()
    with pytest.raises(ValueError, match="(?i)external"):
        register_for_jit_tuning([source], device_management="external")
    assert "forward" not in source.__dict__
    assert not is_externally_managed_module(source)
    assert Patcher._entry_path is None
