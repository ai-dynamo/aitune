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
"""TensorRT Builder module for building TensorRT engines from ONNX models."""

import gc
import logging
from pathlib import Path
from typing import Any

from polygraphy.backend.trt import CreateConfig, Profile, engine_from_network, network_from_onnx_path, save_engine
from wrapt import lazy_import

from aitune.utils.system_monitor import SystemMonitor

trt = lazy_import("tensorrt")

# Setup logger
logger = logging.getLogger(__name__)


class TensorRTBuilder:
    """Class for building TensorRT engines from ONNX models."""

    def __init__(
        self,
        input_onnx_path: Path,
        output_path: Path,
        workspace_size: int | None = None,
        optimization_level: int | None = None,
        compatibility_level: int | None = None,
        timing_cache: Path | None = None,
        profiles: list[Profile] | None = None,
        enable_tf32: bool = True,
    ):
        """Initialize the TensorRT builder.

        Args:
            input_onnx_path: Path to the input ONNX file
            output_path: Path to save the TensorRT engine
            workspace_size: Maximum workspace size in bytes
            cache_dir: Base directory for caching models (will create "tensorrt" subdirectory)
            optimization_level: TensorRT builder optimization level
            compatibility_level: TensorRT hardware compatibility level
            timing_cache: Path to TensorRT timing cache file
            profiles: List of Profile objects for optimization profiles
            enable_tf32: Enable TF32 hardware acceleration
        """
        self.input_onnx_path = input_onnx_path
        self.output_path = output_path
        self.workspace_size = workspace_size
        self.optimization_level = optimization_level
        self.compatibility_level = compatibility_level
        self.timing_cache = timing_cache
        self.profiles = profiles or []
        self.enable_tf32 = enable_tf32
        self.system_monitor = SystemMonitor()

    def build(self) -> Path:
        """Build the TensorRT engine.

        Returns:
            Path to the TensorRT engine
        """
        logger.info("Building TensorRT engine from %s to %s", self.input_onnx_path, self.output_path)

        try:
            # Prepare and validate inputs
            self._validate_onnx_file()

            # Build configuration
            config = self._create_trt_config(self.profiles)

            # Create network, build and save engine
            network = self._create_trt_network()
            engine = self._build_engine_from_network(network, config)
            self._save_engine(engine=engine, path=self.output_path)

            return self.output_path
        except Exception as e:
            self._handle_failed_build(e, self.output_path)
            raise e
        finally:
            gc.collect()

    def _validate_onnx_file(self) -> None:
        """Validate that the ONNX file exists.

        Raises:
            FileNotFoundError: If the ONNX file does not exist
        """
        if not self.input_onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {self.input_onnx_path}")

    def _create_trt_config(self, profiles: list[Profile]) -> Any:
        """Create TensorRT config.

        Args:
            profiles: List of TensorRT optimization profiles

        Returns:
            TensorRT configuration object
        """
        with self.system_monitor.system_stats_context(log_label="TensorRT config creation"):
            # Build configuration using the new method
            config_kwargs = self._build_create_config_kwargs(
                max_workspace_size=self.workspace_size,
                optimization_level=self.optimization_level,
                compatibility_level=self.compatibility_level,
                profiles=profiles,
                timing_cache=self.timing_cache,
                enable_tf32=self.enable_tf32,
            )

            # Create TensorRT config
            config = CreateConfig(**config_kwargs)
            logger.info("TensorRT config created successfully")
            return config

    def _create_trt_network(self) -> Any:
        """Create TensorRT network from ONNX.

        Returns:
            TensorRT network
        """
        logger.info("Creating TensorRT network from ONNX: %s", self.input_onnx_path)
        with self.system_monitor.system_stats_context(log_label="Creating TensorRT network"):
            network = network_from_onnx_path(self.input_onnx_path.as_posix(), strongly_typed=True)
            logger.info("Network created successfully")
            return network

    def _build_engine_from_network(self, network: Any, config: Any) -> Any:
        """Build TensorRT engine from network.

        Args:
            network: TensorRT network
            config: TensorRT configuration

        Returns:
            TensorRT engine
        """
        logger.info("Building TensorRT engine")
        with self.system_monitor.system_stats_context(log_label="Engine building from network"):
            engine = engine_from_network(network=network, config=config)
            logger.info("Engine built successfully")
            return engine

    def _handle_failed_build(self, error: Exception, engine_path: Path) -> None:
        """Handle errors during engine build.

        Args:
            error: The exception that occurred
            engine_path: Path to the engine file
        """
        logger.info("Failed to build TensorRT engine: %s", error)
        if engine_path.exists():
            logger.info("Removing incomplete engine file: %s", engine_path)
            try:
                engine_path.unlink()
            except Exception as delete_error:
                logger.warning("Failed to remove incomplete engine file: %s", delete_error)

    def _build_create_config_kwargs(
        self,
        max_workspace_size,
        optimization_level,
        compatibility_level,
        profiles,
        timing_cache,
        enable_tf32,
    ):
        """Build the kwargs for the CreateConfig constructor.

        Args:
            max_workspace_size: Maximum workspace size
            optimization_level: TensorRT optimization level
            compatibility_level: Hardware compatibility level
            profiles: List of optimization profiles
            timing_cache: Path to timing cache
            enable_tf32: Enable TF32 hardware acceleration

        Returns:
            Dictionary of kwargs for CreateConfig
        """
        try:
            logger.info("Building TensorRT configuration keyword arguments")
            create_config_kwargs = {
                "profiles": profiles,
                "load_timing_cache": timing_cache,
            }

            if optimization_level:
                logger.info("Using optimization level: %s", optimization_level)
                create_config_kwargs["builder_optimization_level"] = optimization_level
            if compatibility_level:
                logger.info("Using compatibility level: %s", compatibility_level)
                create_config_kwargs["hardware_compatibility_level"] = compatibility_level

            if max_workspace_size:
                logger.info("Using workspace size: %s bytes", max_workspace_size)
                create_config_kwargs["memory_pool_limits"] = {
                    trt.MemoryPoolType.WORKSPACE: max_workspace_size,
                }

            if enable_tf32:
                logger.info("Using TF32 hardware acceleration")
                create_config_kwargs["tf32"] = True

            logger.info("TensorRT configuration: %s", create_config_kwargs)
            return create_config_kwargs
        except Exception as e:
            logger.info("Failed to build TensorRT configuration: %s", e)
            raise e

    def _save_engine(self, engine, path: Path):
        """Save the TensorRT engine to a file.

        Args:
            engine: TensorRT engine
            path: Path to save the engine
        """
        try:
            logger.info("Saving TensorRT engine to %s", path)
            with self.system_monitor.system_stats_context(log_label="Saving engine"):
                save_engine(engine=engine, path=path.as_posix())
                logger.info("TensorRT engine built and saved successfully: %s", path)
        except Exception as e:
            logger.error("Failed to save TensorRT engine: %s", e)
            raise e
