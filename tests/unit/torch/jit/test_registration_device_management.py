# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for placement policies on explicitly registered JIT targets."""

from copy import deepcopy
from unittest.mock import Mock

import torch

from aitune.torch import register_for_jit_tuning
from aitune.torch.backend.backend import BuildMode, DummyBackend, ExecutionMode
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.patcher import Patcher, jit_reset
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


class _UnsupportedExternalBackend(DummyBackend):
    _execution_modes = frozenset({ExecutionMode.SINGLE_GPU})
    _build_mode = BuildMode.AHEAD_OF_TIME
    _supports_external_device_management = False


def _configure_dry_run(tmp_path, strategy: Mock) -> None:
    jit_config.mode = JITMode.TUNE_DEFERRED
    jit_config.device = torch.device("cpu")
    jit_config.batch_axis_required = False
    jit_config.min_parameters = -1
    jit_config.dry_run = True
    jit_config.dry_run_failure_probability = 0.0
    jit_config.cache_dir = tmp_path / "jit-cache"
    jit_config.strategy = strategy


def _strategy() -> Mock:
    strategy = Mock()
    strategy.clone.return_value = strategy
    strategy.to_json_dict.return_value = {}
    strategy.backend_results = []
    return strategy


def test_external_registration_preserves_target_tree_but_not_unrelated_modules(tmp_path):
    strategy = _strategy()
    _configure_dry_run(tmp_path, strategy)
    target = _ModuleTree()
    unrelated = torch.nn.Linear(2, 2)
    target.to = Mock(return_value=target)
    target.child.to = Mock(return_value=target.child)
    unrelated.to = Mock(return_value=unrelated)

    register_for_jit_tuning([target], device_management="external")
    move_module_to_device(target, "cpu")
    move_module_to_device(target.child, "cpu")
    move_module_to_device(unrelated, "cpu")

    target.to.assert_not_called()
    target.child.to.assert_not_called()
    unrelated.to.assert_called_once_with("cpu")
    assert is_externally_managed_module(target)
    assert is_externally_managed_module(target.child)
    assert not is_externally_managed_module(unrelated)


def test_external_registration_does_not_mark_deepcopies_as_externally_managed(tmp_path):
    strategy = _strategy()
    _configure_dry_run(tmp_path, strategy)
    target = _ModuleTree()
    register_for_jit_tuning([target], device_management="external")
    patched = Patcher._patched_modules[0]

    patched._restore_original_forward()
    try:
        target_copy = deepcopy(target)
    finally:
        patched._proxy_forward()
    target_copy.to = Mock(return_value=target_copy)

    move_module_to_device(target_copy, "cpu")

    assert not is_externally_managed_module(target_copy)
    target_copy.to.assert_called_once_with("cpu")


def test_external_registration_prevents_offload_and_cleanup(tmp_path, mocker):
    strategy = _strategy()
    _configure_dry_run(tmp_path, strategy)
    target = _ModuleTree()
    target.to = Mock(return_value=target)
    cleanup_memory = mocker.patch("aitune.torch.utils.module.cleanup_memory")
    register_for_jit_tuning([target], device_management="external")

    offload(target, device="meta")

    target.to.assert_not_called()
    cleanup_memory.assert_not_called()


def test_external_policy_preserves_placement_through_successful_dry_run_tuning(tmp_path):
    strategy = _strategy()
    _configure_dry_run(tmp_path, strategy)
    jit_config.device = torch.device("meta")
    module = torch.nn.Linear(2, 2)
    original_class = module.__class__
    module.to = Mock(return_value=module)
    registration = register_for_jit_tuning([module], device_management="external")
    inputs = torch.ones(1, 2)

    module(inputs)
    Patcher.enable_tune_deferred()
    module(inputs)
    result = module(inputs)

    assert registration.all_tuned
    assert registration.state_counts == {"tuned": 1}
    assert module.__class__ is original_class
    assert result.device.type == "cpu"
    module.to.assert_not_called()
    strategy.tune_dry_run.assert_called_once()


def test_default_policy_retains_aitune_device_management(tmp_path):
    strategy = _strategy()
    _configure_dry_run(tmp_path, strategy)
    module = torch.nn.Linear(2, 2)
    original_class = module.__class__
    module.to = Mock(return_value=module)
    registration = register_for_jit_tuning([module])

    module(torch.ones(1, 2))
    Patcher.enable_tune_deferred()
    module(torch.ones(1, 2))

    assert registration.all_tuned
    assert module.__class__ is not original_class
    module.to.assert_called_once_with(torch.device("cpu"))


def test_external_policy_preserves_placement_during_tuning_failure(tmp_path):
    strategy = _strategy()
    strategy.tune_dry_run.side_effect = RuntimeError("build failed")
    _configure_dry_run(tmp_path, strategy)
    module = torch.nn.Linear(2, 2)
    original_class = module.__class__
    module.to = Mock(return_value=module)
    registration = register_for_jit_tuning([module], device_management="external")

    module(torch.ones(1, 2))
    Patcher.enable_tune_deferred()
    module(torch.ones(1, 2))

    assert registration.state_counts == {"eager": 1}
    assert module.__class__ is original_class
    assert not is_externally_managed_module(module)
    module.to.assert_not_called()


def test_jit_reset_removes_external_placement_registration(tmp_path):
    strategy = _strategy()
    _configure_dry_run(tmp_path, strategy)
    module = torch.nn.Linear(2, 2)
    register_for_jit_tuning([module], device_management="external")
    assert is_externally_managed_module(module)

    jit_reset()

    assert not is_externally_managed_module(module)
    assert module.forward.__func__ is torch.nn.Linear.forward
    assert module(torch.ones(1, 2)).shape == (1, 2)


def test_backend_capability_rejects_external_placement_only(tmp_path):
    strategy = _strategy()
    _configure_dry_run(tmp_path, strategy)
    external = torch.nn.Linear(2, 2)
    ordinary = torch.nn.Linear(2, 2)
    register_for_jit_tuning([external], device_management="external")
    backend = _UnsupportedExternalBackend()

    try:
        backend._assert_external_device_management(external)
    except RuntimeError as error:
        assert "external module device management" in str(error)
    else:
        raise AssertionError("unsupported backend accepted an externally managed module")

    backend._assert_external_device_management(ordinary)
    DummyBackend()._assert_external_device_management(external)


def test_backend_build_rejects_unsupported_external_policy_before_build(tmp_path):
    strategy = _strategy()
    _configure_dry_run(tmp_path, strategy)
    module = torch.nn.Linear(2, 2)
    register_for_jit_tuning([module], device_management="external")
    backend = _UnsupportedExternalBackend()
    backend._build = Mock(return_value=backend)

    try:
        backend.build(module, Mock(), [], torch.device("cpu"), tmp_path)
    except RuntimeError as error:
        assert "external module device management" in str(error)
    else:
        raise AssertionError("unsupported backend reached its build implementation")

    backend._build.assert_not_called()
