# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TensorRT Runtime module for running TensorRT engines."""

import logging
from pathlib import Path
from typing import Any

from wrapt import lazy_import

from aitune.global_context import LIBRARY_LOGGING_KEY, global_context
from aitune.torch.backend.tensorrt.tensorrt_engine_info import TensorRTEngineInfo
from aitune.utils.monitoring import annotate

trt = lazy_import("tensorrt")

TRT_ENGINE_FILE_EXTENSION = ".plan"
LOGGING_PREFIX = "[TRT]"

# Setup logger
logger = logging.getLogger(__name__)


def _get_log_level_mapping():
    return {
        logging.DEBUG: trt.Logger.VERBOSE,
        logging.INFO: trt.Logger.INFO,
        logging.WARNING: trt.Logger.WARNING,
        logging.ERROR: trt.Logger.ERROR,
        logging.CRITICAL: trt.Logger.INTERNAL_ERROR,
    }


class PythonTRTLogger(trt.ILogger):
    """TensorRT logger that routes messages through Python's logging system.

    This allows TensorRT C++ output to be captured by Python logging handlers
    and redirected (e.g., to a file) via control_output context manager.
    """

    def __init__(self, severity=trt.Logger.INFO):
        """Initialize with TensorRT severity level."""
        super().__init__()
        self.severity = severity
        self.python_logger = logging.getLogger("tensorrt")

    def log(self, severity, msg):
        """Route TensorRT log messages to Python logging system."""
        # Only log if severity meets threshold
        if severity > self.severity:
            return

        # Map TensorRT severity to Python logging levels
        if severity == trt.Logger.INTERNAL_ERROR:
            self.python_logger.critical("%s %s", LOGGING_PREFIX, msg)
        elif severity == trt.Logger.ERROR:
            self.python_logger.error("%s %s", LOGGING_PREFIX, msg)
        elif severity == trt.Logger.WARNING:
            self.python_logger.warning("%s %s", LOGGING_PREFIX, msg)
        elif severity == trt.Logger.INFO:
            self.python_logger.info("%s %s", LOGGING_PREFIX, msg)
        elif severity == trt.Logger.VERBOSE:
            self.python_logger.debug("%s %s", LOGGING_PREFIX, msg)


class TensorRTRuntime:
    """Class for running TensorRT engines."""

    def __init__(
        self,
        cuda_graph: bool = False,
    ):
        """Initialize the TensorRT runtime.

        Args:
            cuda_graph: Whether to use CUDA graphs for inference
        """
        self.cuda_graph = cuda_graph

    def load_engine(self, engine_path: str | Path) -> bytes:
        """Load the TensorRT engine.

        Args:
            engine_path: Path to the TensorRT engine

        Returns:
            Serialized TensorRT engine
        """
        engine_path = Path(engine_path)
        logger.debug("Loading TensorRT engine from %s", engine_path)

        with annotate("Loading engine file"):
            try:
                # Check if engine exists
                if not engine_path.exists():
                    raise FileNotFoundError(f"TensorRT engine file not found: {engine_path}")

                with open(engine_path, "rb") as f:
                    engine_bytes = f.read()

                if not engine_bytes:
                    raise ValueError(f"TensorRT engine file is empty: {engine_path}")

                logger.debug(
                    "Successfully loaded TensorRT engine from %s (%s MB)",
                    engine_path,
                    len(engine_bytes) / (1024 * 1024),
                )

                return engine_bytes
            except Exception as e:
                logger.debug("Failed to load TensorRT engine: %s", e)
                raise e

    def create_execution_context(self, engine_bytes: bytes) -> tuple[Any, dict, list, list, TensorRTEngineInfo]:
        """Create an execution context from a serialized engine.

        Args:
            engine_bytes: Serialized TensorRT engine

        Returns:
            Tuple containing (context, io_tensors, input_names, output_names, engine_info)
        """
        logger.debug("Creating TensorRT execution context")

        with annotate("Creating execution context"):
            try:
                # Create runtime and deserialize engine
                log_level = global_context.get(LIBRARY_LOGGING_KEY, logger.level)

                # Use Python-based logger that routes through logging system
                # This allows control_output() to capture TensorRT messages to files
                trt_log_severity = _get_log_level_mapping().get(
                    log_level if log_level <= logging.CRITICAL else logging.CRITICAL, trt.Logger.INFO
                )
                trt_logger = PythonTRTLogger(severity=trt_log_severity)

                runtime = trt.Runtime(trt_logger)
                engine = runtime.deserialize_cuda_engine(engine_bytes)
                if not engine:
                    raise RuntimeError("Failed to deserialize engine")

                # Create execution context
                context = engine.create_execution_context()
                if not context:
                    raise RuntimeError("Failed to create execution context")

                # Create io_tensors for execution
                io_tensors = {}
                input_names = []
                output_names = []

                # Map binding indices to names
                for i in range(engine.num_io_tensors):
                    name = engine.get_tensor_name(i)
                    tensor_type = "input" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "output"
                    if tensor_type == "input":
                        input_names.append(name)
                    else:
                        output_names.append(name)
                    dims = engine.get_tensor_shape(name)
                    io_tensors[name] = {
                        "index": i,
                        "type": tensor_type,
                        "shape": dims,
                        "dtype": str(engine.get_tensor_dtype(name)),
                    }

                logger.debug(
                    "Created execution context with %s inputs and %s outputs",
                    len(input_names),
                    len(output_names),
                )

                # Create TensorRTEngineInfo directly from the deserialized engine
                engine_info = TensorRTEngineInfo(engine=engine)

                return context, io_tensors, input_names, output_names, engine_info
            except Exception as e:
                logger.debug("Failed to create execution context: %s", e)
                raise e
