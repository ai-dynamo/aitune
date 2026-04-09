# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Logging configuration for the AITune package."""

import contextlib
import logging
import os
import sys
import traceback
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from aitune.global_context import LIBRARY_LOGGING_KEY, global_context
from aitune.utils.env_vars import CONSOLE_OUTPUT_ENABLE


def setup_logging(
    level: int | str | None = None,
    format_string: str = "%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    log_file: str | Path | None = None,
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


    Args:
        level: Logging level (default: None, uses current root logger level)
        format_string: Format for log messages
        log_file: Optional file path to write logs to
        capture_warnings: If True, warnings will be captured and displayed
    """
    # Configure root logger
    root_logger = logging.getLogger()
    log_file = Path(log_file) if log_file else None

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
        log_dir = log_file.parent
        if log_dir and not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(str(log_file))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Log confirmation
    logging.info("Logging configured - level: %s", logging.getLevelName(root_level))
    if log_file:
        logging.info("Logs are being written to: %s", os.path.abspath(log_file))


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
    reraise: bool = False,
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


class _TeeFile:
    """File-like object that writes to multiple destinations (tee functionality)."""

    def __init__(self, *files):
        """Initialize with multiple file objects to write to."""
        self.files = files

    def write(self, data: str) -> int:
        """Write data to all file objects."""
        for f in self.files:
            f.write(data)
            f.flush()
        return len(data)

    def flush(self) -> None:
        """Flush all file objects."""
        for f in self.files:
            f.flush()

    def isatty(self) -> bool:
        """Return whether this is an interactive stream."""
        # Check if any of the underlying files is a tty
        return any(hasattr(f, "isatty") and f.isatty() for f in self.files)

    def fileno(self) -> int:
        """Return file descriptor of first file."""
        # Return the first file's fileno if available
        for f in self.files:
            if hasattr(f, "fileno"):
                try:
                    return f.fileno()
                except (AttributeError, OSError):
                    continue
        raise OSError("No file descriptor available")

    def readable(self) -> bool:
        """Return whether object supports reading."""
        return False

    def writable(self) -> bool:
        """Return whether object supports writing."""
        return True


def _redirect_low_level_output(target_fd):
    """Redirect low-level file descriptors to target and return cleanup info."""
    save_fds = [os.dup(1), os.dup(2)]
    os.dup2(target_fd, 1)
    os.dup2(target_fd, 2)
    return save_fds


def _restore_low_level_output(save_fds):
    """Restore low-level file descriptors."""
    if save_fds:
        os.dup2(save_fds[0], 1)
        os.dup2(save_fds[1], 2)
        for fd in save_fds:
            os.close(fd)


def _try_redirect_handler(handler, target_file, original_stdout, original_stderr):
    """Try to redirect a single handler to target file.

    Args:
        handler: The logging handler to redirect
        target_file: File object to redirect to
        original_stdout: Original sys.stdout reference
        original_stderr: Original sys.stderr reference

    Returns:
        Tuple of (handler, original_stream) if successful, None otherwise
    """
    if not isinstance(handler, logging.StreamHandler):
        return None
    if handler.stream not in (original_stdout, original_stderr):
        return None
    if not hasattr(handler, "setStream"):
        return None

    try:
        original_stream = handler.stream
        handler.setStream(target_file)
        return (handler, original_stream)
    except (AttributeError, TypeError):
        return None


def _redirect_logging_handlers(target_file, original_stdout, original_stderr):
    """Redirect all logging StreamHandlers to target file.

    Args:
        target_file: File object to redirect to
        original_stdout: Original sys.stdout reference
        original_stderr: Original sys.stderr reference

    Returns:
        List of (handler, original_stream) tuples for restoration
    """
    saved_handler_streams = []
    root_logger = logging.getLogger()

    # Check all logger handlers
    for logger_obj in logging.root.manager.loggerDict.values():
        if isinstance(logger_obj, logging.Logger):
            for handler in logger_obj.handlers:
                result = _try_redirect_handler(handler, target_file, original_stdout, original_stderr)
                if result:
                    saved_handler_streams.append(result)

    # Check root logger handlers
    for handler in root_logger.handlers:
        result = _try_redirect_handler(handler, target_file, original_stdout, original_stderr)
        if result:
            saved_handler_streams.append(result)

    return saved_handler_streams


def _restore_logging_handlers(saved_handler_streams):
    """Restore logging StreamHandlers to their original streams.

    Args:
        saved_handler_streams: List of (handler, original_stream) tuples
    """
    for handler, original_stream in saved_handler_streams:
        handler.setStream(original_stream)


@contextmanager
def control_output(log_file: str | Path | None = None):
    """Silences or redirects stdout, stderr, and logs within the invoked context.

    Args:
        log_file: Optional file path to log output to

    Behavior matrix based on CONSOLE_OUTPUT_ENABLE environment variable and log_file:
        CONSOLE_OUTPUT_ENABLE=False, log_file=None       → Complete suppression (to /dev/null)
        CONSOLE_OUTPUT_ENABLE=False, log_file="file.log" → Redirect to file only, no console
        CONSOLE_OUTPUT_ENABLE=True, log_file=None      → Normal console output
        CONSOLE_OUTPUT_ENABLE=True, log_file="file.log"→ Tee mode: console AND file

    Example usage:
        # Suppress all output
        with control_output():
            print("Not shown anywhere")

        # Redirect output to file only
        with control_output(log_file="output.log"):
            print("Goes to file, not console")
    """
    # If no suppression and no log file, do nothing
    if CONSOLE_OUTPUT_ENABLE and not log_file:
        yield
        return

    # Determine target file
    if log_file:
        log_file = Path(log_file)
        log_dir = log_file.parent
        if log_dir and not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
        target_file = str(log_file)
        mode = "a"
    else:
        target_file = os.devnull
        mode = "w"

    # Save state
    save_fds = None
    tqdm_ctx = None
    saved_handler_streams = []

    # Open target file
    with open(target_file, mode) as f:
        try:
            # Store original stdout/stderr before any redirection
            original_stdout = sys.stdout
            original_stderr = sys.stderr

            if not CONSOLE_OUTPUT_ENABLE:
                # Redirect low-level fds (1,2) - catches C++/Fortran output
                save_fds = _redirect_low_level_output(f.fileno())

                # Redirect logging handlers - critical for Python logging output
                saved_handler_streams = _redirect_logging_handlers(f, original_stdout, original_stderr)

                # Redirect Python-level stdout/stderr to file only
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    yield

            else:
                # CONSOLE_OUTPUT_ENABLE set to True with log_file: tee mode (console AND file)
                # Add a FileHandler for logging (keeps console handlers active)
                file_handler = logging.FileHandler(str(log_file), mode="a")
                # Copy formatter from existing handler if available
                for handler in logging.getLogger().handlers:
                    if isinstance(handler, logging.StreamHandler) and handler.formatter:
                        file_handler.setFormatter(handler.formatter)
                        break
                logging.getLogger().addHandler(file_handler)

                try:
                    # For print() statements, use Tee to write to both console and file
                    tee_stdout = _TeeFile(original_stdout, f)
                    tee_stderr = _TeeFile(original_stderr, f)
                    with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):  # type: ignore[arg-type]
                        yield
                finally:
                    # Remove the temporary file handler
                    logging.getLogger().removeHandler(file_handler)
                    file_handler.close()

        finally:
            if tqdm_ctx:
                tqdm_ctx.__exit__(None, None, None)

            # Restore all redirections
            _restore_logging_handlers(saved_handler_streams)
            _restore_low_level_output(save_fds)
