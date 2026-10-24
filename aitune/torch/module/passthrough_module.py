# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Passthrough module."""

from typing import Any

import torch

from aitune.torch.utils.module import move_module_to_device, move_tensors_to_device


class PassthroughModule:
    """Module that passes through the original module.

    When a device is provided, the module and its inputs are moved to that device. A
    ``None`` device preserves caller-managed module and input placement.

    """

    def __init__(
        self,
        module,
        device: str | torch.device | None,
    ) -> None:
        """Initializes module.

        Args:
                module: module to be tuned.
                device: Device on which the tuned module has to be executed, or ``None``
                    to preserve externally managed placement.
        """
        super().__init__()
        self._forward_call = module.__call__
        self._device = device
        if self._device is not None:
            move_module_to_device(module, self._device)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Native inference on wrapped module."""
        args, kwargs = self._prepare_inputs(*args, **kwargs)
        return self._forward_call(*args, **kwargs)

    def _prepare_inputs(self, *args, **kwargs):
        """Prepare inputs for inplace inference and place them on the same device if are not."""
        if self._device is None:
            return args, kwargs
        return (
            move_tensors_to_device(args, self._device),
            move_tensors_to_device(kwargs, self._device),
        )
