# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspecting and patching modules."""

from logging import getLogger

from aitune.torch.inspecting.module_info import ModuleInfo
from aitune.torch.module.wrapper_module import Module, StrategyList, StrategyMap
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_strategy.tune_strategy import TuneStrategy
from aitune.utils.logging import setup_logging

logger = getLogger(__name__)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def wrap(
    obj: object,
    modules: list[ModuleInfo],
    strategy: TuneStrategy | None = None,
    strategies: StrategyList | StrategyMap | None = None,
) -> object:
    """Wrap provided modules with inspection logic.

    Args:
        obj: Callable object to wrap.
        modules: Dictionary of module names and their corresponding ModuleInfo objects.
        strategy: Strategy to use for patching.
        strategies: Strategies to use for patching.

    Returns:
        Wrapped callable object.
    """
    setup_logging(format_string=LOG_FORMAT)
    MODULE_REGISTRY.clear()
    for module_info in modules:
        logger.info("Wrapping module: %s", module_info.object_path or module_info.name or obj.__class__.__name__)
        if module_info.parent is None:
            return Module(obj, name=obj.__class__.__name__, strategy=strategy, strategies=strategies)

        ait_module = Module(module_info.module, name=module_info.name, strategy=strategy, strategies=strategies)
        module_info.parent.set_wrapped(module_info.name, ait_module)

    return obj
