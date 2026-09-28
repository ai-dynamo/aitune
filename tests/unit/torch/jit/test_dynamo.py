# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for session Dynamo capacity used by explicitly registered JIT targets."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from aitune.torch import register_for_jit_tuning
from aitune.torch.jit import dynamo
from aitune.torch.jit import patched_module as patched_module_module
from aitune.torch.jit import patcher as patcher_module
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.patched_module import PatchedModule
from aitune.torch.jit.patcher import Patcher, jit_reset

_LIMIT_NAMES = (
    "cache_size_limit",
    "recompile_limit",
    "accumulated_cache_size_limit",
    "accumulated_recompile_limit",
)


class _DynamoConfig:
    """Minimal in-memory implementation of TorchDynamo's integer limits."""

    cache_size_limit: int
    recompile_limit: int
    accumulated_cache_size_limit: int
    accumulated_recompile_limit: int

    def __init__(self, **limits: int) -> None:
        for name, value in limits.items():
            setattr(self, name, value)


class _SharedFrameModule(torch.nn.Module):
    """Small module whose instances share one Dynamo frame."""

    def __init__(self, value: int) -> None:
        super().__init__()
        self.value = value

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add an instance-specific constant."""
        return tensor + self.value


def _strategy(num_backends: int) -> Mock:
    strategy = Mock()
    strategy.to_json_dict.return_value = {"backends": [{} for _ in range(num_backends)]}
    return strategy


def test_capacity_reserves_build_and_runtime_routes_until_reset(monkeypatch):
    config = _DynamoConfig(**dict.fromkeys(_LIMIT_NAMES, 2))
    monkeypatch.setattr(dynamo, "dynamo_config", config)

    dynamo.reserve_dynamo_recompile_capacity(owner=1, route_count=4, strategy=_strategy(num_backends=3))

    # Four routes need three candidate-build variants plus one final inference variant each.
    assert {name: getattr(config, name) for name in _LIMIT_NAMES} == dict.fromkeys(_LIMIT_NAMES, 16)

    jit_reset()
    assert {name: getattr(config, name) for name in _LIMIT_NAMES} == dict.fromkeys(_LIMIT_NAMES, 2)


def test_capacity_grows_monotonically_and_restores_original_limits(monkeypatch):
    original_limits = {
        "cache_size_limit": 1,
        "recompile_limit": 2,
        "accumulated_cache_size_limit": 3,
        "accumulated_recompile_limit": 4,
    }
    config = _DynamoConfig(**original_limits)
    monkeypatch.setattr(dynamo, "dynamo_config", config)

    dynamo.reserve_dynamo_recompile_capacity(owner=1, route_count=2, strategy=_strategy(num_backends=1))
    assert all(getattr(config, name) == 4 for name in _LIMIT_NAMES)

    dynamo.reserve_dynamo_recompile_capacity(owner=1, route_count=4, strategy=_strategy(num_backends=2))
    assert all(getattr(config, name) == 12 for name in _LIMIT_NAMES)

    dynamo.reserve_dynamo_recompile_capacity(owner=1, route_count=1, strategy=_strategy(num_backends=1))
    assert all(getattr(config, name) == 12 for name in _LIMIT_NAMES)

    dynamo.restore_dynamo_recompile_capacity()
    assert {name: getattr(config, name) for name in _LIMIT_NAMES} == original_limits


def test_capacity_restores_after_every_active_target_releases(monkeypatch):
    config = _DynamoConfig(**dict.fromkeys(_LIMIT_NAMES, 2))
    monkeypatch.setattr(dynamo, "dynamo_config", config)

    dynamo.reserve_dynamo_recompile_capacity(owner=10, route_count=2, strategy=_strategy(num_backends=1))
    dynamo.reserve_dynamo_recompile_capacity(owner=11, route_count=3, strategy=_strategy(num_backends=1))
    assert all(getattr(config, name) == 6 for name in _LIMIT_NAMES)

    dynamo.release_dynamo_recompile_capacity(10)
    assert all(getattr(config, name) == 6 for name in _LIMIT_NAMES)

    dynamo.release_dynamo_recompile_capacity(11)
    assert all(getattr(config, name) == 2 for name in _LIMIT_NAMES)


def test_first_successful_tuned_inference_releases_target_capacity(monkeypatch):
    module = torch.nn.Linear(2, 2)
    patched = PatchedModule(module, explicit_head=True)
    expected = torch.ones(1, 2)
    patched._wrapper = Mock(return_value=expected)
    patched._update_state(patched_module_module.ModuleState.TUNED)
    release = Mock()
    monkeypatch.setattr(patched_module_module, "release_dynamo_recompile_capacity", release)

    assert module(torch.zeros(1, 2)) is expected

    release.assert_called_once_with(patched._id)


def test_capacity_is_held_until_every_recorded_jit_route_runs(monkeypatch):
    patched = PatchedModule(torch.nn.Linear(2, 2), explicit_head=True)
    first_route = Mock()
    second_route = Mock()
    patched._pending_dynamo_runtime_routes = {first_route, second_route}
    release = Mock()
    monkeypatch.setattr(patched_module_module, "release_dynamo_recompile_capacity", release)

    patched._mark_dynamo_runtime_route_complete(first_route)
    release.assert_not_called()

    patched._mark_dynamo_runtime_route_complete(second_route)
    release.assert_called_once_with(patched._id)


def test_failed_selected_inference_keeps_target_capacity_for_retry(monkeypatch):
    module = torch.nn.Linear(2, 2)
    patched = PatchedModule(module, explicit_head=True)
    patched._wrapper = Mock(side_effect=RuntimeError("lazy compile failed"))
    patched._update_state(patched_module_module.ModuleState.TUNED)
    release = Mock()
    monkeypatch.setattr(patched_module_module, "release_dynamo_recompile_capacity", release)

    with pytest.raises(RuntimeError, match="lazy compile failed"):
        module(torch.zeros(1, 2))

    release.assert_not_called()


def test_capacity_preserves_external_limit_updates_across_later_reservations(monkeypatch):
    config = _DynamoConfig(**dict.fromkeys(_LIMIT_NAMES, 8))
    monkeypatch.setattr(dynamo, "dynamo_config", config)

    dynamo.reserve_dynamo_recompile_capacity(owner=1, route_count=4, strategy=_strategy(num_backends=3))
    assert all(getattr(config, name) == 16 for name in _LIMIT_NAMES)

    for name in _LIMIT_NAMES:
        setattr(config, name, 32)

    dynamo.reserve_dynamo_recompile_capacity(owner=1, route_count=16, strategy=_strategy(num_backends=3))
    assert all(getattr(config, name) == 64 for name in _LIMIT_NAMES)

    dynamo.restore_dynamo_recompile_capacity()
    assert all(getattr(config, name) == 32 for name in _LIMIT_NAMES)


def test_capacity_leaves_higher_limits_unchanged_without_patching(monkeypatch):
    original_limits = dict.fromkeys(_LIMIT_NAMES, 100)
    config = _DynamoConfig(**original_limits)
    monkeypatch.setattr(dynamo, "dynamo_config", config)

    dynamo.reserve_dynamo_recompile_capacity(owner=1, route_count=2, strategy=_strategy(num_backends=2))
    dynamo.restore_dynamo_recompile_capacity()

    assert {name: getattr(config, name) for name in _LIMIT_NAMES} == original_limits


def test_capacity_only_changes_limits_available_in_the_installed_torch(monkeypatch):
    config = _DynamoConfig(cache_size_limit=1, accumulated_cache_size_limit=20)
    monkeypatch.setattr(dynamo, "dynamo_config", config)

    dynamo.reserve_dynamo_recompile_capacity(owner=1, route_count=5, strategy=_strategy(num_backends=2))
    assert config.cache_size_limit == 15
    assert config.accumulated_cache_size_limit == 20
    assert not hasattr(config, "recompile_limit")
    assert not hasattr(config, "accumulated_recompile_limit")

    dynamo.restore_dynamo_recompile_capacity()
    assert config.cache_size_limit == 1
    assert config.accumulated_cache_size_limit == 20


def test_explicit_route_count_uses_recorded_graphs_and_ignores_automatic_wrappers():
    explicit_recorded = PatchedModule(torch.nn.Linear(2, 2), explicit_head=True)
    explicit_recorded._wrapper = SimpleNamespace(graph_specs=(object(), object(), object()))
    explicit_unrecorded = PatchedModule(torch.nn.Linear(2, 2), explicit_head=True)
    automatic = PatchedModule(torch.nn.Linear(2, 2))
    automatic._wrapper = SimpleNamespace(graph_specs=(object(), object(), object(), object()))
    Patcher._patched_modules.extend([explicit_recorded, explicit_unrecorded, automatic])

    assert Patcher.explicit_route_count() == 4

    explicit_recorded._wrapper = SimpleNamespace()
    explicit_recorded._unpatch()
    explicit_unrecorded._wrapper = SimpleNamespace(graph_specs=(object(), object()))

    assert Patcher.explicit_route_count() == 5


def test_explicit_tuning_reserves_session_capacity_but_automatic_tuning_does_not(monkeypatch):
    explicit = PatchedModule(torch.nn.Linear(2, 2), explicit_head=True)
    automatic = PatchedModule(torch.nn.Linear(2, 2))
    Patcher._patched_modules.extend([explicit, automatic])
    strategy = _strategy(num_backends=2)
    calls = []

    def capacity(owner, route_count, selected_strategy, *, detect_graph_breaks):
        calls.append((owner, route_count, selected_strategy, detect_graph_breaks))

    monkeypatch.setattr(patched_module_module, "reserve_dynamo_recompile_capacity", capacity)
    monkeypatch.setattr(patched_module_module.config, "detect_graph_breaks", True)

    explicit._reserve_dynamo_tuning_capacity(strategy)
    automatic._reserve_dynamo_tuning_capacity(strategy)

    assert calls == [(explicit._id, 1, strategy, True)]


def test_explicit_dry_run_does_not_reserve_dynamo_capacity(tmp_path, monkeypatch):
    strategy = _strategy(num_backends=2)
    strategy.clone.return_value = strategy
    strategy.backend_results = []
    jit_config.mode = JITMode.TUNE_DEFERRED
    jit_config.device = torch.device("cpu")
    jit_config.batch_axis_required = False
    jit_config.min_parameters = -1
    jit_config.cache_dir = tmp_path / "jit-cache"
    jit_config.dry_run = True
    jit_config.dry_run_failure_probability = 0.0
    jit_config.strategy = strategy
    reserve = Mock()
    monkeypatch.setattr(patched_module_module, "reserve_dynamo_recompile_capacity", reserve)
    module = torch.nn.Linear(2, 2)
    register_for_jit_tuning([module])
    inputs = torch.ones(1, 2)

    module(inputs)
    Patcher.enable_tune_deferred()
    module(inputs)

    reserve.assert_not_called()
    strategy.tune_dry_run.assert_called_once()


def test_session_capacity_covers_detection_build_and_post_tuning_routes():
    compiled_graphs = 0

    def detector_compiler(graph_module, _example_inputs):
        nonlocal compiled_graphs
        compiled_graphs += 1
        return graph_module.forward

    def backend_compiler(graph_module, _example_inputs):
        nonlocal compiled_graphs
        compiled_graphs += 1
        return graph_module.forward

    modules = [_SharedFrameModule(value) for value in range(3)]
    available_limits = {name: 1 for name in _LIMIT_NAMES if hasattr(dynamo.dynamo_config, name)}
    torch._dynamo.reset()
    try:
        with dynamo.dynamo_config.patch(**available_limits):
            # Detection, one backend candidate, and the post-device-patch inference variant need all nine routes.
            dynamo.reserve_dynamo_recompile_capacity(
                owner=1,
                route_count=3,
                strategy=_strategy(num_backends=1),
                detect_graph_breaks=True,
            )
            inputs = torch.tensor(1)
            detectors = [torch.compile(module, backend=detector_compiler, fullgraph=True) for module in modules]
            assert [compiled(inputs).item() for compiled in detectors] == [1, 2, 3]

            optimized = [torch.compile(module, backend=backend_compiler, fullgraph=True) for module in modules]
            assert [compiled(inputs).item() for compiled in optimized] == [1, 2, 3]

            for index, module in enumerate(modules):
                module.__class__ = type(
                    f"PatchedSharedFrameModule{index}",
                    (_SharedFrameModule,),
                    {"device": torch.device("cpu")},
                )

            assert [compiled(inputs).item() for compiled in optimized] == [1, 2, 3]
            assert compiled_graphs == 9

            dynamo.restore_dynamo_recompile_capacity()
            assert all(getattr(dynamo.dynamo_config, name) == 1 for name in available_limits)
    finally:
        dynamo.restore_dynamo_recompile_capacity()
        torch._dynamo.reset()


def test_reset_restores_capacity_when_report_finalization_fails(monkeypatch):
    config = _DynamoConfig(**dict.fromkeys(_LIMIT_NAMES, 1))
    monkeypatch.setattr(dynamo, "dynamo_config", config)
    dynamo.reserve_dynamo_recompile_capacity(owner=1, route_count=3, strategy=_strategy(num_backends=1))
    assert all(getattr(config, name) == 6 for name in _LIMIT_NAMES)

    with monkeypatch.context() as failure:
        Patcher._session_mode = JITMode.TUNE_EAGER
        failure.setattr(patcher_module, "has_active_report", lambda: True)
        failure.setattr(
            patcher_module,
            "report_tune_run_end",
            Mock(side_effect=RuntimeError("report failed")),
        )

        with pytest.raises(RuntimeError, match="report failed"):
            jit_reset()

    assert all(getattr(config, name) == 1 for name in _LIMIT_NAMES)
    assert Patcher._session_mode is None


def test_reset_preserves_failed_wrapper_for_retry_after_partial_unpatch(monkeypatch):
    config = _DynamoConfig(**dict.fromkeys(_LIMIT_NAMES, 1))
    monkeypatch.setattr(dynamo, "dynamo_config", config)
    wrappers = [PatchedModule(torch.nn.Linear(2, 2), explicit_head=True) for _ in range(2)]
    modules = [wrapper.__wrapped__ for wrapper in wrappers]
    original_forwards = [wrapper._original_forward for wrapper in wrappers]
    Patcher._patched_modules.extend(wrappers)
    Patcher._session_mode = JITMode.TUNE_EAGER
    Patcher._explicit_route_counts.update({wrapper._id: 1 for wrapper in wrappers})
    module_counter = PatchedModule.module_counter
    dynamo.reserve_dynamo_recompile_capacity(owner=1, route_count=2, strategy=_strategy(num_backends=1))

    with monkeypatch.context() as failure:
        failure.setattr(patcher_module, "has_active_report", lambda: False)
        failure.setattr(wrappers[1], "_unpatch", Mock(side_effect=RuntimeError("unpatch failed")))

        with pytest.raises(RuntimeError, match="unpatch failed"):
            jit_reset()

    assert all(getattr(config, name) == 1 for name in _LIMIT_NAMES)
    assert Patcher._session_mode == JITMode.TUNE_EAGER
    assert Patcher._patched_modules == [wrappers[1]]
    assert Patcher._explicit_route_counts == {wrapper._id: 1 for wrapper in wrappers}
    assert PatchedModule.module_counter == module_counter
    assert modules[0].forward is original_forwards[0]
    assert modules[1].forward is not original_forwards[1]

    jit_reset()

    assert Patcher._patched_modules == []
    assert Patcher._session_mode is None
    assert Patcher._explicit_route_counts == {}
    assert PatchedModule.module_counter == 0
    assert all(module.forward is original for module, original in zip(modules, original_forwards, strict=True))
