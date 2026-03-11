# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AITune exceptions."""

import pathlib


class AITuneError(Exception):
    """Base exception for AITune exceptions."""

    def __init__(self, message: str, log_path: pathlib.Path | None = None):
        """Initialize exception object.

        Args:
            message: An error message
            log_path: A path to log file to store logs
        """
        self._message = message
        self._log_path = log_path

    def __str__(self):
        """Convert exception object to string.

        Returns:
            Error message of exception
        """
        return self._message

    @property
    def message(self) -> str:
        """Get the exception message.

        Returns:
            The message associated with this exception, or None if no message.
        """
        return self._message

    @property
    def log_path(self) -> pathlib.Path:
        """Get the log file path.

        Returns:
            The path to file where logs are stored, or None if no path.
        """
        return self._log_path


class AITuneUserInputError(AITuneError):
    """AITuneUserInputError exception.

    Raised when user provided input (model, data or configuration option) is invalid.
    """

    pass
