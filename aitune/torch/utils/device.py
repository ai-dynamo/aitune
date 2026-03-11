# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simple CUDA utilities."""

import re

import torch


def get_device(device: str | torch.device) -> torch.device:
    """Get the current CUDA device.

    Returns:
        str: The current CUDA device in format 'cuda:<device_id>'.
    """
    device_str = None
    if isinstance(device, torch.device):
        if device.type == "cuda":
            device_str = f"{device.type}:{device.index or 0}"
        elif device.type in ["cpu", "meta"]:
            device_str = device.type
        else:
            raise ValueError(f"Invalid device: {device}. Expected 'cuda', 'cpu' or 'meta'")
    elif isinstance(device, str):
        if device in ["cpu", "meta"]:
            device_str = device
        else:
            pattern = r"^cuda:(\d+)$"
            match = re.match(pattern, device)
            if match:
                index = int(match.group(1))
                if index < 0 or index > 127:
                    raise ValueError(f"Invalid device index: {index}. Expected 0-127.")
                device_str = device

            pattern = r"^cuda$"
            match = re.match(pattern, device)
            if match:
                device_str = "cuda:0"

    if device_str is None:
        raise ValueError(f"Invalid device: {device}. Expected 'cpu', 'cuda' or 'cuda:<device_id>' in range 0-127")

    return torch.device(device_str)
