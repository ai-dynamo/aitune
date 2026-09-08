# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hashing utilities for the AITune package."""

import hashlib
from pathlib import Path

_HASH_CHUNK_SIZE = 1024 * 1024


def hash_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def hash_string(s: str) -> str:
    """Hash a string using SHA-256."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
