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
"""Utility functions for PyTorch modules."""

import torch.nn as nn


def count_parameters(module: nn.Module) -> str:
    """Counts the total number of parameters and returns it in a human-readable format.

    Args:
        module: The PyTorch module to count parameters for.

    Returns:
        A string representation of the parameter count in human-readable format
        (e.g., "1.2M", "500K", "100").

    Examples:
        >>> import torch.nn as nn
        >>> module = nn.Linear(1000, 100)
        >>> count_parameters(module)
        '100.1K'
    """
    num_params = sum(p.numel() for p in module.parameters())

    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.1f}B"
    elif num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.1f}M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.1f}K"
    else:
        return f"{num_params}"
