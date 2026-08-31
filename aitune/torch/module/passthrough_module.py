# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Passthrough module."""

from typing import Any

from aitune.torch.utils.module import move_module_to_device, move_tensors_to_device


class PassthroughModule:
    """Module that passes through the original module.

    Before passing the data this module moves the data to the same device as the module.

    """

    def __init__(
        self,
        module,
        device: str,
    ) -> None:
        """Initializes module.

        Args:
                module: module to be tuned.
                device: Device on which tuned module has to be executed.
        """
        super().__init__()
        self._forward_call = module.__call__
        self._device = device
        move_module_to_device(module, self._device)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Native inference on wrapped module."""
        args, kwargs = self._prepare_inputs(*args, **kwargs)
        return self._forward_call(*args, **kwargs)

    def _prepare_inputs(self, *args, **kwargs):
        """Prepare inputs for inplace inference and place them on the same device if are not."""
        return (
            move_tensors_to_device(args, self._device),
            move_tensors_to_device(kwargs, self._device),
        )
