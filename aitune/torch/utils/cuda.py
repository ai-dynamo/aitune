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
"""Simple CUDA utilities."""

import re

import torch


def assert_is_available():
    """Assert that CUDA is available.

    Code should be testable even if CUDA is not available.

    By providing a assertion function, we allow mocking the CUDA availability.

    Raises:
        RuntimeError: If CUDA is not available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your CUDA installation.")


def is_available():
    """Check if CUDA is available.

    Returns:
        bool: True if CUDA is available.
    """
    return torch.cuda.is_available()


def synchronize():
    """Synchronize all CUDA devices if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_device(device: str | torch.device) -> str:
    """Get the current CUDA device.

    Returns:
        str: The current CUDA device in format 'cuda:<device_id>'.
    """
    if isinstance(device, torch.device):
        if device.type != "cuda":
            raise ValueError("Device must be 'cuda' type")
        return f"{device.type}:{device.index or 0}"
    elif isinstance(device, str):
        pattern = r"^cuda:(\d+)$"
        match = re.match(pattern, device)
        if match:
            return device

        pattern = r"^cuda$"
        match = re.match(pattern, device)
        if match:
            return "cuda:0"

    raise ValueError("device must be 'cuda' or in format 'cuda:<device_id>'")


def set_device(device: str | torch.device):
    """Set the current CUDA device.

    Args:
        device: Device string or torch.device object. If string is 'cuda', it will be converted to 'cuda:0'.
    """
    device = get_device(device)
    torch.cuda.set_device(device)
