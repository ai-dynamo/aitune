# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from aitune.torch.jit.config import JITMode, config
from aitune.torch.jit.patcher import Patcher

config.mode = JITMode.INSPECT
config.max_depth_level = 7
Patcher.patch_torch()


def save_report(path: Path | str, model_name: str):
    """Save inspection report to html file."""
    from aitune.torch.jit.inspect_module import InspectModule

    InspectModule.save_report(path, model_name)
