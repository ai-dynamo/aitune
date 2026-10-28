# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspecting and patching modules."""

from collections.abc import Callable
from logging import getLogger
from typing import Any, TypeVar, cast

import torch

from aitune.torch.inspecting.module_info import ModuleInfo
from aitune.torch.module.wrapper_module import Module, StrategyList, StrategyMap
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_strategy.resolver import StrategyOption
from aitune.utils.logging import setup_logging

logger = getLogger(__name__)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


T = TypeVar("T", bound=torch.nn.Module | Module | Callable[..., Any])


def wrap(
    obj: T,
    modules: list[ModuleInfo],
    strategy: StrategyOption | None = None,
    strategies: StrategyList | StrategyMap | None = None,
) -> T | Module:
    """Wrap provided modules with inspection logic.

    Args:
        obj: Callable object to wrap.
        modules: Dictionary of module names and their corresponding ModuleInfo objects.
        strategy: Strategy to use for patching.
        strategies: Strategies to use for patching.

    Returns:
        A `Module` proxy if the root module is selected, otherwise `obj` with its submodules wrapped in place.
    """
    setup_logging(format_string=LOG_FORMAT)
    MODULE_REGISTRY.clear()
    for module_info in modules:
        logger.info("Wrapping module: %s", module_info.object_path or module_info.name or obj.__class__.__name__)
        if module_info.parent is None:
            return Module(
                cast(torch.nn.Module, obj), name=obj.__class__.__name__, strategy=strategy, strategies=strategies
            )

        ait_module = Module(module_info.module, name=module_info.name, strategy=strategy, strategies=strategies)
        module_info.parent.set_wrapped(module_info.name, ait_module)

    return obj
