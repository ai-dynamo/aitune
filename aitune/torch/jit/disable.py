# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module for disabling JIT tuning.

If you import this module automatic JIT tuning will be turned off.

Example:
    >>> import aitune.torch.jit.disable
    >>> # JIT tuning is disabled
"""

from aitune.torch.jit.patcher import Patcher

Patcher.unpatch_torch()
