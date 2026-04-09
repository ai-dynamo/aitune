# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Environment variables for AITune."""

import os
from pathlib import Path


def get_bool_env_variable(env_variable: str, default: bool) -> bool:
    """Get a boolean environment variable.

    Args:
        env_variable: The name of the environment variable.
        default: The default value if the environment variable is not set.

    Returns:
        The value of the environment variable.
    """
    value = os.environ.get(env_variable)
    if value is None:
        return default
    return value in ["1", "true", "True", "yes", "Yes", "YES"]


CONSOLE_OUTPUT_ENABLE = get_bool_env_variable("AITUNE_CONSOLE_OUTPUT", False)
HARDWARE_METRICS_ENABLED = get_bool_env_variable("AITUNE_HARDWARE_METRICS", False)
NVTX_ENABLE = get_bool_env_variable("NVTX_ENABLE", False)

_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
AITUNE_CACHE_DIR = Path(os.environ.get("AITUNE_CACHE_DIR", _CACHE_DIR / "aitune"))
AITUNE_JIT_CACHE_DIR = Path(os.environ.get("AITUNE_JIT_CACHE_DIR", _CACHE_DIR / "aitune.jit"))
