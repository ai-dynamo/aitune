# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hashing utilities for the AITune package."""

import hashlib


def hash_string(s: str) -> str:
    """Hash a string using SHA-256."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
