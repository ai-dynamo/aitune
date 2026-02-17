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
"""Torch tuning module."""

# configuring some variables before importing other modules
from aitune.torch.config import aitune_cache_dir, config  # noqa: I001

from aitune.torch.checkpoint.local_torch_storage import LocalTorchStorage
from aitune.torch.dataloader import DataLoaderFactory
from aitune.torch.inspecting import inspect, wrap
from aitune.torch.module import Module
from aitune.torch.tune_strategy import FirstWinsStrategy, HighestThroughputStrategy, OneBackendStrategy, TuneStrategy
from aitune.torch.tuning import load, save, tune
from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.patched_module import PatchedModule
from aitune.torch.jit.patcher import patch_for_jit_tuning

__all__ = [
    "aitune_cache_dir",
    "config",
    "jit_config",
    "inspect",
    "wrap",
    "tune",
    "load",
    "save",
    "Module",
    "PatchedModule",
    "TuneStrategy",
    "OneBackendStrategy",
    "FirstWinsStrategy",
    "HighestThroughputStrategy",
    "LocalTorchStorage",
    "DataLoaderFactory",
    "patch_for_jit_tuning",
]

if config.enable_hf_integrations:
    import aitune.torch.integrations.hugging_face  # noqa: F401
