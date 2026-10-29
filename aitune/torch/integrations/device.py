# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device reporting for modules containing tuned submodules."""

import itertools

import torch
import torch.nn as nn

from aitune.torch.module.wrapper_module import Module, ModuleState


def get_tuned_execution_device(module: nn.Module) -> torch.device | None:
    """Return the execution device of the tuned submodule owning the first parameter or buffer of a module.

    Args:
        module: Module whose first parameter or buffer determines its reported device.

    Returns:
        The device of the nearest tuned AITune module containing that tensor, or None if there is none.

    Examples:
        >>> get_tuned_execution_device(torch.nn.Linear(2, 2)) is None
        True
    """
    tensor_name, _ = next(itertools.chain(module.named_parameters(), module.named_buffers()))
    # e.g. "encoder.conv_in.weight" is owned by the submodule at ["encoder", "conv_in"]
    owner_path = tensor_name.split(".")[:-1]
    # get_submodule() returns wrappers, unlike named_modules(); check the nearest ancestor first.
    for depth in range(len(owner_path), 0, -1):
        submodule = module.get_submodule(".".join(owner_path[:depth]))
        if isinstance(submodule, Module) and submodule.state == ModuleState.TUNED:
            return submodule.device
    return None


def patch_device_property(cls: type) -> None:
    """Make the `device` property of a library base class report the execution device of tuned submodules.

    Diffusers and Transformers report the device of a module's first parameter or buffer. After AOT tuning,
    parameters of tuned submodules are offloaded to meta, so their parents would report the meta device.

    Args:
        cls: Class defining a `device` property. Classes without one are left unchanged.
    """
    original = cls.__dict__.get("device")
    if not isinstance(original, property):
        return

    def device(self: nn.Module) -> torch.device:
        result = original.__get__(self)
        if result.type != "meta":
            return result
        return get_tuned_execution_device(self) or result

    cls.device = property(device, doc=original.__doc__)
