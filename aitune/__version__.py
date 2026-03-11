# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import version

try:
    __version__ = version("aitune")
except Exception:
    __version__ = "0.0.0+unknown"
