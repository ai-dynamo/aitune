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
"""Logging configuration for the AITune package."""

import logging
import os
import sys
import traceback
import warnings
from collections.abc import Callable
from contextlib import contextmanager

from aitune.global_context import LIBRARY_LOGGING_KEY, global_context


def setup_logging(
    level: int | str | None = None,
    format_string: str = "%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    log_file: str | None = None,
    capture_warnings: bool = True,
):
    """Configure logging for the AITune package.

    This function sets up the root logger with appropriate handlers and formatting.
    Call this at the start of your application to ensure all logs are displayed.

    Example usage:
    from aitune import setup_logging, set_module_level
    import logging

    # Configure with a log file and custom format
    setup_logging(
        level=logging.DEBUG,
        format_string="%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
        log_file="aitune.log",
    )

    # Set specific modules to different levels
    set_module_level("aitune.torch.backend.tensorrt", logging.DEBUG)  # Verbose TensorRT logs
    set_module_level("aitune.utils.system_monitor", logging.INFO)     # Normal system logs


    Args:
        level: Logging level (default: None, uses current root logger level)
        format_string: Format for log messages
        log_file: Optional file path to write logs to
        capture_warnings: If True, warnings will be captured and displayed
    """
    # Configure root logger
    root_logger = logging.getLogger()

    # Set level if provided
    if level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        root_logger.setLevel(level)

    root_level = root_logger.level

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Capture warnings for all loggers
    if capture_warnings:
        logging.captureWarnings(True)
        # Configure the warning logger to use the same level as root
        warning_logger = logging.getLogger("py.warnings")
        warning_logger.setLevel(root_level)

    # Clear existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Log confirmation
    logging.info("Logging configured - level: %s", logging.getLevelName(root_level))
    if log_file:
        logging.info("Logs are being written to: %s", os.path.abspath(log_file))

    if root_level == logging.DEBUG:
        enable_gpu_memory_logging()


def set_module_level(module_name: str, level: int | str):
    """Set logging level for a specific module.

    Args:
        module_name: Name of the module (e.g., 'aitune.torch.backend')
        level: Logging level (e.g., 'DEBUG', 'INFO', 'WARNING')
    """
    # Convert string level to numeric value if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    logging.info("Set logging level for %s to %s", module_name, logging.getLevelName(level))


def enable_gpu_memory_logging():
    """Enable detailed GPU memory logging in the SystemMonitor.

    This sets the logging level for the SystemMonitor module to DEBUG,
    allowing system statistics logs to be displayed. By default, the
    SystemMonitor logs at DEBUG level, but those logs won't be visible
    unless the module's logger level is also set to DEBUG.
    """
    set_module_level("aitune.utils.system_monitor", logging.DEBUG)


@contextmanager
def libraries_logging(disabled: bool, exceptions: list[str] | None = None):
    """Suppress logging for all packages except the specified ones.

    Args:
        disabled: If False, libraries logs will be suppressed.
        exceptions: List of package names to exclude from suppression
    """
    if not disabled:
        yield
        return

    if exceptions is None:
        exceptions = ["aitune"]

    original_levels = {}

    # Get all existing loggers
    all_loggers = list(logging.root.manager.loggerDict.keys())

    level = logging.CRITICAL + 1
    root_level = logging.root.level
    try:
        global_context.set(LIBRARY_LOGGING_KEY, level)
        for logger_name in all_loggers:
            # Skip the specified package
            if any(logger_name.startswith(exception) for exception in exceptions):
                continue

            logger = logging.getLogger(logger_name)
            original_levels[logger_name] = logger.level
            logger.setLevel(level)

        # WARNING: colored is not installed, color will not be used
        warnings.filterwarnings("ignore", message=".*'colored' module is not installed.*")

        yield
    finally:
        # Restore original levels
        for logger_name, original_level in original_levels.items():
            logging.getLogger(logger_name).setLevel(original_level)

        global_context.set(LIBRARY_LOGGING_KEY, root_level)
        warnings.resetwarnings()


def log(msg: str, *args, sink: Callable = logging.info, depth: int = 0):
    """Log a message with indentation.

    Args:
        depth: Number of levels to indent
        *args: Arguments to pass to the sink function
        msg: Message to log
        sink: Function to use for logging
    """
    sink("  " * depth + msg, *args)


def log_exception_details(
    logger: logging.Logger,
    exception: Exception,
    message: str,
    level: int = logging.ERROR,
    reraise: bool = True,
    reraise_as: Exception | None = None,
):
    """Log detailed exception information in a standardized format.

    This function provides consistent and comprehensive exception logging
    across the AITune codebase, including exception type, message, and
    full traceback information.

    Args:
        logger: Logger instance to use for logging
        exception: The exception that was caught
        message: Custom error message describing the context
        level: Logging level to use (default: ERROR)
        reraise: Whether to re-raise the exception after logging (default: True)
        reraise_as: Optional exception to raise instead of the original (default: None)

    Raises:
        Exception: Re-raises the original exception or the specified reraise_as exception

    Example:
        try:
            # Some operation that might fail
            risky_operation()
        except Exception as e:
            log_exception_details(
                logger=logger,
                exception=e,
                message="Failed to perform risky operation",
                reraise_as=RuntimeError(f"Operation failed: {e}")
            )
    """
    # Log the main error message with error icon
    logger.log(level, "❌ %s", message)

    # Log exception type and details with info icons
    logger.log(level, "🔍 Exception type: %s", type(exception).__name__)
    logger.log(level, "💬 Exception details: %s", str(exception))

    # Log full traceback with stack trace icon
    logger.log(level, "📋 Full traceback:\n%s", traceback.format_exc())

    # Re-raise if requested
    if reraise:
        if reraise_as is not None:
            raise reraise_as from exception
        else:
            raise exception
