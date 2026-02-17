# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test the patcher functions."""

from unittest.mock import Mock

import torch

from aitune.torch.jit.patcher import Patcher, jit_reset, patch_for_jit_tuning, prepare_for_jit_tuning


def test_jit_reset():
    """Test jit_reset function."""
    Patcher.patch_torch()
    torch.nn.Linear(10, 5)
    assert len(Patcher._patched_modules) == 1
    jit_reset()
    assert len(Patcher._patched_modules) == 0
    assert len(Patcher._intercepted_classes) == 0
    torch.nn.Linear(10, 5)
    assert len(Patcher._patched_modules) == 0
    assert len(Patcher._intercepted_classes) == 0


def test_prepare_for_tuning():
    """Test prepare_for_tuning context manager."""
    pre_module = torch.nn.Linear(10, 5)  # noqa: F841
    with prepare_for_jit_tuning():
        module = torch.nn.Linear(10, 5)
    after_module = torch.nn.Linear(10, 5)  # noqa: F841
    assert len(Patcher._patched_modules) == 1
    patched_module = Patcher._patched_modules[0]
    assert patched_module.__wrapped__ == module


def test_patch_decorator():
    """Test prepare_for_tuning context manager."""

    @patch_for_jit_tuning
    def create_module():
        module = torch.nn.Linear(10, 5)
        return module

    pre_module = torch.nn.Linear(10, 5)  # noqa: F841
    module = create_module()
    after_module = torch.nn.Linear(10, 5)  # noqa: F841
    assert len(Patcher._patched_modules) == 1
    patched_module = Patcher._patched_modules[0]
    assert patched_module.__wrapped__ == module


def test_is_allowed_to_tune():
    """Test is_allowed_to_tune function."""
    assert Patcher._is_allowed_to_tune(torch.nn.Linear(10, 5))
    assert not Patcher._is_allowed_to_tune(Mock(spec=torch._dynamo.eval_frame.OptimizedModule))


def test_intercepted_classes():
    """Test intercepted_classes function."""

    @patch_for_jit_tuning
    def create_module():
        module = torch.nn.Linear(10, 5)
        return module

    _ = create_module()
    assert Patcher.intercepted_classes() == ["torch.nn.modules.linear.Linear"]
