# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspecting and patching modules."""

from collections.abc import Callable
from logging import getLogger
from typing import Any, TypeVar

import torch

from aitune.torch.call_tree.call_tree import CallTree
from aitune.torch.call_tree.weights.sharing import shared_weights_with_ids
from aitune.torch.inspecting.module_info import ModuleInfo
from aitune.torch.module.wrapper_module import Module, StrategyList, StrategyMap
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_strategy.resolver import StrategyOption
from aitune.torch.utils.module import keep_on_device, release_kept_on_device
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
        modules: Modules to wrap, usually chosen from ``ait.inspect(...).get_modules()``. If that inspection
            recorded a call tree (``call_tree=True``), weights of a wrapped module that code outside it reads,
            checks the device of, or moves stay on the device when tuning offloads the module (see
            ``aitune.torch.call_tree.shared_weights_of``), and each kept weight is logged. Wrapping a module again
            replaces the weights it keeps; a weight stays kept while any wrapped module keeps it. A ``ModuleInfo``
            without ``inspection``, such as one built by hand, keeps nothing.
        strategy: Strategy to use for patching.
        strategies: Strategies to use for patching.

    Returns:
        A `Module` proxy if the root module is selected, otherwise `obj` with its submodules wrapped in place.
    """
    setup_logging(format_string=LOG_FORMAT)
    MODULE_REGISTRY.clear()
    for module_info in modules:
        logger.info("Wrapping module: %s", module_info.object_path or module_info.name or obj.__class__.__name__)
        module = module_info.module if module_info.parent is not None else obj
        if isinstance(module, torch.nn.Module):
            release_kept_on_device(module)
            call_tree = module_info.inspection.call_tree if module_info.inspection is not None else None
            if call_tree is not None:
                _keep_shared_weights(
                    call_tree, module, module_info.object_path or module_info.name or type(module).__name__
                )
        if module_info.parent is None:
            assert module_info.module is obj, "Root ModuleInfo must hold the wrapped object"
            return Module(module_info.module, name=obj.__class__.__name__, strategy=strategy, strategies=strategies)

        ait_module = Module(module_info.module, name=module_info.name, strategy=strategy, strategies=strategies)
        module_info.parent.set_wrapped(module_info.name, ait_module)

    return obj


def _keep_shared_weights(call_tree: CallTree, module: torch.nn.Module, name: str) -> None:
    """Keep the weights of ``module`` that something outside it still uses or moves on the device; log them."""
    try:
        weights = shared_weights_with_ids(call_tree, module)
    except ValueError:
        logger.warning("Cannot check %s for shared weights: it is not part of the recorded model", name)
        return
    params = {id(p): p for p in (*module.parameters(), *module.buffers())}
    if weights:
        keep_on_device((params[i] for _, i in weights if i in params), owner=module)
        users = []
        for kind, label in (("read", "read"), ("device", "device checked"), ("move", "moved")):
            by = sorted({a.by for w, _ in weights for a in w.accesses if a.kind == kind})
            if by:
                users.append(f"{label} by {', '.join(by)}")
        logger.info(
            "Keeping %d %s (%.1f MB) of %s on the device after tuning: %s",
            len(weights),
            "weight" if len(weights) == 1 else "weights",
            sum(w.bytes for w, _ in weights) / 1e6,
            name,
            "; ".join(users),
        )
    for weight, _ in weights:
        logger.debug(
            "Keeping %s of %s: %s",
            weight.name,
            name,
            ", ".join(f"{a.kind} by {a.by}" for a in weight.accesses),
        )
