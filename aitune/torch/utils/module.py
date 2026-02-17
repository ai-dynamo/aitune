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
"""PyTorch module utilities for parameter counting, device management, and memory offloading."""

import inspect
from collections.abc import Callable
from typing import Optional

import torch
import torch.nn as nn

from aitune.torch.utils.memory import cleanup_memory


def format_num_parameters(num: int) -> str:
    """Formats a number into human-readable format with appropriate suffixes.

    Args:
        num: The number to format.

    Returns:
        A string representation of the number in human-readable format
        (e.g., "1.2B", "500M", "100K", "50").

    Examples:
        >>> format_num_parameters(1_200_000_000)
        '1.2B'
        >>> format_num_parameters(500_000)
        '500.0K'
        >>> format_num_parameters(50)
        '50'
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return f"{num}"


def count_parameters(module: nn.Module) -> int:
    """Counts the total number of parameters and returns the count as an integer.

    Args:
        module: The PyTorch module to count parameters for.

    Returns:
       The total number of parameters as an integer (e.g., 1200000, 500000, 100).

    Examples:
        >>> import torch.nn as nn
        >>> module = nn.Linear(1000, 100)
        >>> count_parameters(module)
        100100
    """
    num_params = sum(p.numel() for p in module.parameters())
    return num_params


def get_forward_arguments_names(forward_func: Callable) -> tuple[list[str], list[str]]:
    """Get the forward_func signature arguments names.

    Args:
        forward_func: PyTorch module forward function
    Returns:
        Tuples of forward positional only arguments and forward keyword arguments (and positional that could be passed as kwargs)
    """
    forward_signature = inspect.signature(forward_func)
    params_list = list(forward_signature.parameters.values())

    if len(params_list) == 0:
        raise ValueError(f"Forward function '{forward_func.__name__}' has no parameters!")

    allowed_kinds = {inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    forward_args = [p.name for p in params_list if p.kind == inspect.Parameter.POSITIONAL_ONLY]
    forward_kwargs = [p.name for p in params_list if p.kind in allowed_kinds]

    return forward_args, forward_kwargs


def get_module_device(module: nn.Module) -> Optional["torch.device"]:
    """Get the device of the given module.

    Args:
        module: Module to get the device of.

    Returns:
        The device of module based on parameters. If not parameters, returns None.
    """
    try:
        return next(module.parameters()).device
    except StopIteration:
        return None


def offload(model: nn.Module, device: str | torch.device = "meta") -> None:
    """Offload model to meta device and freeing all memory.

    Args:
        model: Model to offload and destroy
        device: Device to offload to
    """
    # Step 1: Move model to meta device
    model.to(device)

    # Step 2: Memory cleanup
    cleanup_memory()
