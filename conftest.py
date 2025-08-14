# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
"""Pytest fixtures both for unit and integration tests and docttest for production code."""

import os

import pytest
import torch

from aitune.torch.jit.patcher import jit_reset
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.utils.cuda import is_available as is_cuda_available


@pytest.fixture(autouse=True)
def module_registry_cleanup():
    """Cleans up the module registry after each test."""
    try:
        yield
    finally:
        MODULE_REGISTRY.clear()


@pytest.fixture(autouse=True)
def aitune_cache_dir(mocker, tmp_path):
    """Sets cache dir to temporary directory."""
    cache_dir = tmp_path / "aitune_cache"
    mocker.patch("aitune.torch.config.DEFAULT_CACHE_DIR", cache_dir)
    return cache_dir


@pytest.fixture
def torch_device():
    """Returns the device specified in AITUNE_TESTS_USE_DEVICE environment variable if set,
    otherwise returns CUDA if available, or CPU as fallback.
    """  # noqa: D205
    user_device = os.environ.get("AITUNE_TESTS_USE_DEVICE")

    if user_device is not None:
        assert is_cuda_available(), "CUDA is not available, but AITUNE_TESTS_USE_DEVICE is set to CUDA"
        return torch.device(user_device)

    return torch.device("cuda" if is_cuda_available() else "cpu")


@pytest.fixture(autouse=True)
def jit_cleanup():
    """Reset patcher state before and after each test."""
    jit_reset()
    try:
        yield
    finally:
        jit_reset()
