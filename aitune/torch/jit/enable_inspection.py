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
"""Module for enabling JIT inspection.

If you import this module, all torch models will be inspected automatically.

Example:
    >>> import aitune.torch.jit.enable_inspection as inspection
    >>> import torch
    >>> model = torch.nn.Linear(10, 5)
    >>> x = model.forward(torch.randn(10))
    >>> x = model.forward(torch.randn(10))
    >>> # Save report to html file
    >>> inspection.save_report("linear_net.html", "Linear net")
"""

from pathlib import Path

from aitune.torch.jit.config import config
from aitune.torch.jit.patcher import Patcher

config.inspect_mode = True
config.max_depth_level = 7
Patcher.patch_torch()


def save_report(path: Path | str, model_name: str):
    """Save inspection report to html file."""
    from aitune.torch.jit.inspect_module import InspectModule

    InspectModule.save_report(path, model_name)
