# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Passthrough module."""

from typing import Any

from aitune.torch.module.sample_metadata import SampleMetadata


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
        module.to(self._device)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Native inference on wrapped module."""
        args, kwargs = self._prepare_inputs(*args, **kwargs)
        return self._forward_call(*args, **kwargs)

    def _prepare_inputs(self, *args, **kwargs):
        """Prepare inputs for inplace inference and place them on the same device if are not."""
        sample_metadata = SampleMetadata.from_inputs(args, kwargs)

        for locator, tensor_spec in sample_metadata.tensor_data:
            if tensor_spec.name.startswith("args"):
                args = locator.set_value(args, locator.get_value(args).to(self._device))
            else:
                kwargs = locator.set_value(kwargs, locator.get_value(kwargs).to(self._device))

        return args, kwargs
