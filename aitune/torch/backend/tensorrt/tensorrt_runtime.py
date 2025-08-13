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
"""TensorRT Runtime module for running TensorRT engines."""

import logging
from pathlib import Path
from typing import Any

import tensorrt as trt

from aitune.global_context import LIBRARY_LOGGING_KEY, global_context
from aitune.torch.backend.tensorrt.tensorrt_engine_info import TensorRTEngineInfo
from aitune.utils.system_monitor import SystemMonitor

TRT_ENGINE_FILE_EXTENSION = ".plan"

# Setup logger
logger = logging.getLogger(__name__)

LOG_LEVEL_MAPPING = {
    logging.DEBUG: trt.Logger.VERBOSE,
    logging.INFO: trt.Logger.INFO,
    logging.WARNING: trt.Logger.WARNING,
    logging.ERROR: trt.Logger.ERROR,
    logging.CRITICAL: trt.Logger.INTERNAL_ERROR,
}


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
        self.system_monitor = SystemMonitor()

    def load_engine(self, engine_path: str | Path) -> bytes:
        """Load the TensorRT engine.

        Args:
            engine_path: Path to the TensorRT engine

        Returns:
            Serialized TensorRT engine
        """
        engine_path = Path(engine_path)
        logger.debug("Loading TensorRT engine from %s", engine_path)

        with self.system_monitor.system_stats_context(log_label="Loading engine file"):
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

        with self.system_monitor.system_stats_context(log_label="Creating execution context"):
            try:
                # Create runtime and deserialize engine
                log_level = global_context.get(LIBRARY_LOGGING_KEY, logger.level)
                if log_level > logging.CRITICAL:
                    trt_log_severity = trt.Logger.INTERNAL_ERROR
                else:
                    trt_log_severity = LOG_LEVEL_MAPPING.get(log_level, trt.Logger.INFO)

                runtime = trt.Runtime(trt.Logger(trt_log_severity))
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
                        "dtype": trt.nptype(engine.get_tensor_dtype(name)),
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
