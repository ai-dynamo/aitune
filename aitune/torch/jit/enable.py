# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
