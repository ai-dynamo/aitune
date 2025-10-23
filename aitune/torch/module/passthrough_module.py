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
        device: str | None = None,
    ) -> None:
        """Initializes module.

        Args:
                module: module to be tuned.
                device: Device on which tuned module has to be executed.
        """
        super().__init__()
        self._forward_call = module.__call__
        self._device = device
        module.to(device)

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
