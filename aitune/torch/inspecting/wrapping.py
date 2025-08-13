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
        if module_info.name is None:
            logger.info("Wrapping module: %s", obj.__class__.__name__)
            return Module(obj, name=obj.__class__.__name__, strategy=strategy, strategies=strategies)

        parts = module_info.name.split(".")
        module = obj
        for part in parts[:-1]:
            module = getattr(module, part)

        logger.info("Wrapping module: %s [%s]", module_info.name, module_info.module_type)
        setattr(
            module,
            parts[-1],
            Module(module_info.module, name=module_info.name, strategy=strategy, strategies=strategies),
        )

    return obj
