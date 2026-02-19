# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
"""Module for enabling JIT tuning.

If you import this module, all torch models will be tuned automatically.

Example:
    >>> import aitune.torch.jit.enable
    >>> import torch
    >>> model = torch.nn.Linear(10, 5)
    >>> x = model.forward(torch.randn(10))
    >>> x = model.forward(torch.randn(10))
    >>> # model is tuned
"""

from aitune.torch.jit.patcher import Patcher

Patcher.patch_torch()


def wrapt_auto_enable(_):
    """Entry point for enabling JIT tuning by autowrapt."""
    import logging

    logging.warning("Enabling JIT tuning")

    Patcher.patch_torch()
