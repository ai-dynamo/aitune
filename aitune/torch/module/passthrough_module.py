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

import torch

from aitune.torch.module.recording_module import INPUT_METADATA_PREFIX
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
        sample = (*args, kwargs)
        sample_metadata = SampleMetadata.from_sample(sample, prefix=INPUT_METADATA_PREFIX)

        input_sample = {}
        for n, t in sample_metadata.flatten_sample(sample).items():
            if isinstance(t, torch.Tensor) and t.device != self._device:
                t = t.to(self._device)
            input_sample[n] = t

        unflatten_inputs = sample_metadata.unflatten_sample(input_sample, wrap_input=True)
        if isinstance(unflatten_inputs[-1], dict):
            device_args, device_kwargs = unflatten_inputs[:-1], unflatten_inputs[-1]
        else:
            device_args, device_kwargs = unflatten_inputs, {}

        return device_args, device_kwargs
