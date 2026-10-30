# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diffusers integrations."""

import torch

try:
    from diffusers import ContextParallelConfig

    from aitune.torch.integrations import register_distributed_module_detector

    def _is_context_parallel_module(module: torch.nn.Module) -> bool:
        for child in module.modules():
            parallel_config = getattr(child, "_parallel_config", None)
            context_parallel_config = getattr(parallel_config, "context_parallel_config", parallel_config)
            if isinstance(context_parallel_config, ContextParallelConfig):
                return True
        return False

    register_distributed_module_detector("diffusers_context_parallel", _is_context_parallel_module)
except ImportError:
    pass

try:
    from diffusers import ModelMixin

    from aitune.torch.integrations.device import patch_device_property

    # report the execution device of tuned submodules instead of meta
    patch_device_property(ModelMixin)
except ImportError:
    pass
