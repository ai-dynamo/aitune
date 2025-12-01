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
from pathlib import Path

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


def get_file_size(path: str | Path) -> int:
    """Get the size of a file.

    Args:
        path: The path to the file.

    Returns:
        The size of the file in bytes.
    """
    return Path(path).stat().st_size


def format_file_size(size_bytes: int) -> str:
    """Formats a file size in bytes into human-readable format.

    Args:
        size_bytes: The file size in bytes.

    Returns:
        A string representation of the file size in human-readable format
        (e.g., "1.5 GB", "500.0 MB", "100.0 KB", "50 B").

    Examples:
        >>> format_file_size(1_500_000_000)
        '1.4 GB'
        >>> format_file_size(500_000)
        '488.3 KB'
        >>> format_file_size(50)
        '50 B'
    """
    if size_bytes >= 1_073_741_824:  # 1024^3
        return f"{size_bytes / 1_073_741_824:.1f} GB"
    elif size_bytes >= 1_048_576:  # 1024^2
        return f"{size_bytes / 1_048_576:.1f} MB"
    elif size_bytes >= 1_024:  # 1024
        return f"{size_bytes / 1_024:.1f} KB"
    else:
        return f"{size_bytes} B"
