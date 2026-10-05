# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for session Dynamo capacity used by explicitly registered JIT targets."""

from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from aitune.torch import register_for_jit_tuning
from aitune.torch.backend.backend import BuildMode
from aitune.torch.jit import dynamo
from aitune.torch.jit import patched_module as patched_module_module
from aitune.torch.jit import patcher as patcher_module
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.patched_module import PatchedModule
from aitune.torch.jit.patcher import Patcher, jit_reset
from aitune.torch.module.forward_signature import ForwardSignature
from aitune.torch.module.passthrough_module import PassthroughModule
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.module.tuned_module import TunedModule
from aitune.torch.tune_strategy.tune_strategy import DummyTuneStrategy
from aitune.torch.utils.module import is_externally_managed_module

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


def _tuned_wrapper(patched, inputs, *, build_mode=BuildMode.JUST_IN_TIME):
    signature = ForwardSignature.from_callable(patched._original_forward)
    backends = OrderedDict()
    for value in inputs:
        route = SampleMetadata.from_inputs(signature.normalize((value,), {}).arguments)
        backend = Mock(build_mode=build_mode)
        backend.infer.side_effect = lambda tensor: tensor + 1
        backends[route] = backend
    return TunedModule(backends, module_name="test", forward_signature=signature)


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
    value = torch.zeros(1, 2)
    patched._wrapper = _tuned_wrapper(patched, [value])
    patched._pending_dynamo_runtime_routes = set(patched._wrapper.backends)
    patched._update_state(patched_module_module.ModuleState.TUNED)
    release = Mock()
    monkeypatch.setattr(patched_module_module, "release_dynamo_recompile_capacity", release)
    route_for_inputs = Mock(wraps=patched._wrapper.route_for_inputs)
    monkeypatch.setattr(patched._wrapper, "route_for_inputs", route_for_inputs)

    torch.testing.assert_close(module(value), value + 1)
    torch.testing.assert_close(module(value), value + 1)
    torch.testing.assert_close(module(value), value + 1)

    release.assert_called_once_with(patched._id)
    route_for_inputs.assert_called_once()
    assert not patched._pending_dynamo_runtime_routes


def test_capacity_is_held_until_every_recorded_jit_route_succeeds_after_retry(monkeypatch):
    config = _DynamoConfig(**dict.fromkeys(_LIMIT_NAMES, 1))
    monkeypatch.setattr(dynamo, "dynamo_config", config)
    module = torch.nn.Linear(2, 2)
    patched = PatchedModule(module, explicit_head=True)
    first_value, second_value = torch.zeros(1, 2), torch.zeros(2)
    patched._wrapper = _tuned_wrapper(patched, [first_value, second_value])
    first_route, second_route = patched._wrapper.backends
    patched._pending_dynamo_runtime_routes = {first_route, second_route}
    patched._update_state(patched_module_module.ModuleState.TUNED)
    dynamo.reserve_dynamo_recompile_capacity(patched._id, route_count=2, strategy=_strategy(num_backends=1))
    release = Mock(wraps=dynamo.release_dynamo_recompile_capacity)
    monkeypatch.setattr(patched_module_module, "release_dynamo_recompile_capacity", release)

    torch.testing.assert_close(module(first_value), first_value + 1)
    assert patched._pending_dynamo_runtime_routes == {second_route}
    release.assert_not_called()
    assert all(getattr(config, name) == 4 for name in _LIMIT_NAMES)

    backend = patched._wrapper.backends[second_route]
    backend.infer.side_effect = RuntimeError("lazy compile failed")
    with pytest.raises(RuntimeError, match="lazy compile failed"):
        module(second_value)
    assert module.forward is not patched._original_forward
    assert patched._pending_dynamo_runtime_routes == {second_route}
    release.assert_not_called()
    assert all(getattr(config, name) == 4 for name in _LIMIT_NAMES)

    backend.infer.side_effect = lambda tensor: tensor + 1
    torch.testing.assert_close(module(second_value), second_value + 1)
    assert not patched._pending_dynamo_runtime_routes
    release.assert_called_once_with(patched._id)
    assert all(getattr(config, name) == 1 for name in _LIMIT_NAMES)

    torch.testing.assert_close(module(first_value), first_value + 1)
    torch.testing.assert_close(module(second_value), second_value + 1)
    release.assert_called_once_with(patched._id)


@pytest.mark.parametrize("explicit_head", [True, False])
@pytest.mark.parametrize("build_mode", [BuildMode.JUST_IN_TIME, BuildMode.AHEAD_OF_TIME])
def test_steady_state_tuned_inference_skips_capacity_bookkeeping(monkeypatch, explicit_head, build_mode):
    module = torch.nn.Linear(2, 2)
    patched = PatchedModule(module, explicit_head=explicit_head)
    value = torch.zeros(1, 2)
    patched._wrapper = _tuned_wrapper(patched, [value], build_mode=build_mode)
    patched._update_state(patched_module_module.ModuleState.TUNED)
    release = Mock()
    monkeypatch.setattr(patched_module_module, "release_dynamo_recompile_capacity", release)
    route_for_inputs = Mock(side_effect=AssertionError("steady-state route lookup"))
    monkeypatch.setattr(patched._wrapper, "route_for_inputs", route_for_inputs)

    torch.testing.assert_close(module(value), value + 1)
    torch.testing.assert_close(module(value), value + 1)

    backend = next(iter(patched._wrapper.backends.values()))
    backend.infer.side_effect = RuntimeError("inference failed")
    with pytest.raises(RuntimeError, match="inference failed"):
        module(value)
    assert module.forward is not patched._original_forward
    backend.infer.side_effect = lambda tensor: tensor + 1
    torch.testing.assert_close(module(value), value + 1)

    route_for_inputs.assert_not_called()
    release.assert_not_called()


@pytest.mark.parametrize("explicit_head", [True, False])
def test_passthrough_inference_skips_capacity_bookkeeping(monkeypatch, explicit_head):
    module = torch.nn.Linear(2, 2)
    value = torch.ones(1, 2)
    expected = module(value)
    patched = PatchedModule(module, explicit_head=explicit_head)
    patched._wrapper = PassthroughModule(module, device=torch.device("cpu"))
    patched._update_state(patched_module_module.ModuleState.TUNED)
    release = Mock()
    monkeypatch.setattr(patched_module_module, "release_dynamo_recompile_capacity", release)
    mark_complete = Mock()
    monkeypatch.setattr(patched, "_mark_dynamo_runtime_route_complete", mark_complete)

    torch.testing.assert_close(module(value), expected)
    torch.testing.assert_close(module(value), expected)

    mark_complete.assert_not_called()
    release.assert_not_called()


def test_capacity_uses_custom_strategy_prepared_for_current_module(monkeypatch):
    class ModuleDependentStrategy(DummyTuneStrategy):
        def __init__(self):
            super().__init__()
            self._backends = []

        def _configure_for_module(self, module):
            self._backends = [f"candidate-{index}" for index in range(module.candidate_count)]

        def to_json_dict(self):
            return {"backends": list(self._backends)}

    prior_module = _SharedFrameModule(1)
    prior_module.candidate_count = 2
    original_strategy = ModuleDependentStrategy()
    original_strategy._configure_for_module(prior_module)
    jit_config.strategy = original_strategy
    current_module = _SharedFrameModule(2)
    current_module.candidate_count = 4
    patched = PatchedModule(current_module, explicit_head=True)
    Patcher._patched_modules.append(patched)
    config = _DynamoConfig(**dict.fromkeys(_LIMIT_NAMES, 1))
    monkeypatch.setattr(dynamo, "dynamo_config", config)

    prepared_strategy = patched_module_module._build_strategy(current_module)
    patched._reserve_dynamo_tuning_capacity(prepared_strategy)

    assert prepared_strategy is not original_strategy
    assert len(prepared_strategy.to_json_dict()["backends"]) == 4
    assert len(original_strategy.to_json_dict()["backends"]) == 2
    assert all(getattr(config, name) == 5 for name in _LIMIT_NAMES)


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
        Patcher._session_report = Mock()
        failure.setattr(
            patcher_module,
            "report_tune_run_end",
            Mock(side_effect=RuntimeError("report failed")),
        )

        with pytest.raises(RuntimeError, match="report failed"):
            jit_reset()

    assert all(getattr(config, name) == 1 for name in _LIMIT_NAMES)
    assert Patcher._session_mode is None
    assert Patcher._session_report is None


def test_reset_preserves_failed_wrapper_for_retry_after_partial_unpatch(monkeypatch):
    config = _DynamoConfig(**dict.fromkeys(_LIMIT_NAMES, 1))
    monkeypatch.setattr(dynamo, "dynamo_config", config)
    wrappers = [
        PatchedModule(
            torch.nn.Linear(2, 2),
            explicit_head=True,
            device_management="external",
        )
        for _ in range(2)
    ]
    modules = [wrapper.__wrapped__ for wrapper in wrappers]
    original_forwards = [wrapper._original_forward for wrapper in wrappers]
    Patcher._patched_modules.extend(wrappers)
    Patcher._session_mode = JITMode.TUNE_EAGER
    Patcher._explicit_route_counts.update({wrapper._id: 1 for wrapper in wrappers})
    module_counter = PatchedModule.module_counter
    dynamo.reserve_dynamo_recompile_capacity(owner=1, route_count=2, strategy=_strategy(num_backends=1))
    assert all(is_externally_managed_module(module) for module in modules)

    with monkeypatch.context() as failure:
        failure.setattr(patcher_module, "has_active_report", lambda: False)
        failure.setattr(wrappers[1], "_unpatch", Mock(side_effect=RuntimeError("unpatch failed")))

        with pytest.raises(RuntimeError, match="unpatch failed"):
            jit_reset()

    assert all(getattr(config, name) == 1 for name in _LIMIT_NAMES)
    assert Patcher._session_mode == JITMode.TUNE_EAGER
    assert Patcher._cleanup_pending
    assert Patcher._patched_modules == [wrappers[1]]
    assert Patcher._explicit_route_counts == {wrapper._id: 1 for wrapper in wrappers}
    assert PatchedModule.module_counter == module_counter
    assert modules[0].forward is original_forwards[0]
    assert not is_externally_managed_module(modules[0])
    assert is_externally_managed_module(modules[1])
    assert modules[1].forward is not original_forwards[1]
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        register_for_jit_tuning([modules[1]], device_management="external")

    jit_reset()

    assert Patcher._patched_modules == []
    assert Patcher._session_mode is None
    assert not Patcher._cleanup_pending
    assert Patcher._explicit_route_counts == {}
    assert PatchedModule.module_counter == 0
    assert all(module.forward is original for module, original in zip(modules, original_forwards, strict=True))
    assert not any(is_externally_managed_module(module) for module in modules)


def test_old_cached_forward_does_not_release_new_session_capacity(monkeypatch):
    config = _DynamoConfig(**dict.fromkeys(_LIMIT_NAMES, 1))
    monkeypatch.setattr(dynamo, "dynamo_config", config)
    jit_config.mode = JITMode.TUNE_DEFERRED
    jit_config.device = torch.device("cpu")
    module = torch.nn.Linear(2, 2)
    value = torch.ones(1, 2)
    expected = module(value)
    register_for_jit_tuning([module])
    old_wrapper = Patcher._patched_modules[0]
    cached_forward = module.forward
    jit_reset()

    registration = register_for_jit_tuning([module])
    new_wrapper = Patcher._patched_modules[0]
    assert new_wrapper._id == old_wrapper._id
    dynamo.reserve_dynamo_recompile_capacity(owner=new_wrapper._id, route_count=3, strategy=_strategy(num_backends=1))

    torch.testing.assert_close(cached_forward(value), expected)

    assert all(getattr(config, name) == 6 for name in _LIMIT_NAMES)
    assert registration.state_counts == {"init": 1}
    assert registration.reports[0].call_count == 0
    assert Patcher._patched_modules == [new_wrapper]


@pytest.mark.parametrize("settled_state", ["skipped", "eager"])
def test_repeated_settled_registration_preserves_historical_routes_without_new_capacity(
    tmp_path, monkeypatch, settled_state
):
    class FailingStrategy(DummyTuneStrategy):
        def _tune(self, *args):
            raise RuntimeError("no backend can tune this target")

    jit_config.mode = JITMode.TUNE_DEFERRED
    jit_config.device = torch.device("cpu")
    jit_config.batch_axis_required = False
    jit_config.min_parameters = -1
    jit_config.cache_dir = tmp_path / "jit-cache"
    jit_config.strategy = FailingStrategy()
    module = torch.nn.Linear(2, 2)
    inputs = (torch.ones(1, 2), torch.ones(2))
    first = register_for_jit_tuning([module])
    module_id = first.reports[0].module_id
    module_counter = PatchedModule.module_counter
    reserve = Mock(wraps=dynamo.reserve_dynamo_recompile_capacity)
    monkeypatch.setattr(patched_module_module, "reserve_dynamo_recompile_capacity", reserve)

    for value in inputs:
        module(value)
    assert len(first.reports[0].graphs) == 2

    if settled_state == "skipped":
        jit_config.skip_modules = ["Linear"]
    else:
        Patcher.enable_tune_deferred()
    module(inputs[0])
    assert first.state_counts == {settled_state: 1}
    assert Patcher._patched_modules == []
    assert Patcher.explicit_route_count() == 2
    reservations_before_reuse = reserve.call_count

    for _ in range(3):
        repeated = register_for_jit_tuning([module])
        assert repeated.state_counts == {settled_state: 1}
        assert repeated.reports[0].module_id == module_id
        assert PatchedModule.module_counter == module_counter
        assert Patcher.explicit_route_count() == 2
        assert Patcher._patched_modules == []
        for value in inputs:
            module(value)
        assert reserve.call_count == reservations_before_reuse

    assert first.state_counts == {settled_state: 1}
