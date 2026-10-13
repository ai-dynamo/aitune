# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cached JIT forward references remain callable through ordinary skip and eager fallback."""

import pytest
import torch

from aitune.torch import register_for_jit_tuning
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.patched_module import ModuleState
from aitune.torch.jit.patcher import Patcher


@pytest.fixture(autouse=True)
def deferred_cpu_session():
    jit_config.mode = JITMode.TUNE_DEFERRED
    jit_config.device = torch.device("cpu")
    jit_config.batch_axis_required = False
    jit_config.min_parameters = -1


@pytest.mark.parametrize("state", [ModuleState.EAGER, ModuleState.DETACHED])
def test_cached_forward_stays_eager_after_permanent_unpatch(state):
    module = torch.nn.Linear(2, 2)
    value = torch.ones(1, 2)
    expected = module(value)
    register_for_jit_tuning([module])
    cached_forward = module.forward
    wrapper = Patcher._patched_modules[0]
    wrapper._update_state(state)

    wrapper._unpatch()

    torch.testing.assert_close(cached_forward(input=value), expected)

    assert wrapper._state == state
    assert wrapper._call_count == 0
    assert Patcher._patched_modules == []
    assert module.forward is wrapper._original_forward


def test_cached_forward_stays_callable_after_parameter_filter_skips_module():
    module = torch.nn.Linear(2, 2)
    value = torch.ones(1, 2)
    expected = module(value)
    jit_config.min_parameters = 100
    registration = register_for_jit_tuning([module])
    cached_forward = module.forward
    module(value)

    torch.testing.assert_close(cached_forward(input=value), expected)

    assert registration.state_counts == {"skipped": 1}
    assert registration.reports[0].call_count == 1
    assert Patcher._patched_modules == []
