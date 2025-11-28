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
"""Utility functions for path and filename manipulation."""

import re

MAX_FILENAME_BYTES = 80


def sanitize_filename(name: str, max_bytes: int = MAX_FILENAME_BYTES) -> str:
    """Sanitize a string for use as a filename.

    Args:
        name: The original string to sanitize.
        max_bytes: Maximum size in bytes for the UTF-8 encoded filename.

    Returns:
        A sanitized filename.
    """
    # Replace special characters (e.g. emojis, spaces) with underscores
    safe = re.sub(r"[^a-zA-Z0-9.\-_]+", "_", name.strip())

    if not safe.strip():
        return "unnamed"

    # Prevent consecutive dots and leading/trailing dots
    safe = re.sub(r"\.{2,}", "_", safe)
    safe = safe.strip(".")

    # Truncate to max_bytes
    encoded = safe.encode("utf-8")
    if len(encoded) > max_bytes:
        encoded = encoded[:max_bytes]
        safe = encoded.decode("utf-8", errors="ignore")

    return safe
