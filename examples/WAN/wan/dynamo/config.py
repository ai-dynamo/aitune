# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared configuration helpers for the WAN Dynamo backend and client."""

import os
from pathlib import Path
from typing import Any

import yaml

from ..defaults import DEFAULT_HEIGHT, DEFAULT_NUM_FRAMES, DEFAULT_WIDTH


def load_config() -> dict[str, Any]:
    """Load the configured WAN Dynamo settings."""
    config_path = Path(os.environ.get("AITUNE_EXAMPLE_CONFIG_PATH", "config.yaml"))
    with config_path.open() as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"WAN configuration must be a mapping: {config_path}")
    return config


def validate_generation_shape(settings: dict[str, Any], *, width: int, height: int, num_frames: int) -> None:
    """Reject request shapes that do not match the statically tuned checkpoint."""
    supported_width = int(settings.get("width", DEFAULT_WIDTH))
    supported_height = int(settings.get("height", DEFAULT_HEIGHT))
    supported_num_frames = int(settings.get("num_frames", DEFAULT_NUM_FRAMES))
    supported_shape = (supported_width, supported_height, supported_num_frames)
    requested_shape = (width, height, num_frames)
    if requested_shape != supported_shape:
        raise ValueError(
            "WAN checkpoint supports only the configured generation shape "
            f"{supported_width}x{supported_height} with {supported_num_frames} frames; "
            f"requested {width}x{height} with {num_frames} frames. "
            "Retune for the requested shape and update the serving configuration."
        )
