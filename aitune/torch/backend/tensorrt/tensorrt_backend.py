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
"""TensorRT backend."""

import contextlib
import copy
import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nvtx
import torch
import torch.nn as nn
from polygraphy.logger import G_LOGGER

from aitune.torch.backend.backend import Backend, BackendConfig, BackendState
from aitune.torch.backend.tensorrt.onnx_autocast import ONNXAutoCast, ONNXAutoCastConfig
from aitune.torch.backend.tensorrt.onnx_exporter import ONNXExporter
from aitune.torch.backend.tensorrt.onnx_quantization import ONNXQuantizationConfig, ONNXQuantizer
from aitune.torch.backend.tensorrt.tensorrt_builder import TensorRTBuilder
from aitune.torch.backend.tensorrt.tensorrt_profile import TensorRTProfile
from aitune.torch.backend.tensorrt.tensorrt_runtime import TensorRTRuntime
from aitune.torch.backend.tensorrt.torch_output_allocator import TorchOutputAllocator
from aitune.torch.backend.tensorrt.torch_quantization import TorchQuantizationConfig, TorchQuantizer
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.utils.cuda import set_device as cuda_set_device
from aitune.utils.system_monitor import SystemMonitor

G_LOGGER.use_python_logging_system = True

# File extension constants
TRT_ENGINE_FILE_EXTENSION = ".plan"
ONNX_FILE_EXTENSION = ".onnx"


# Setup logger
logger = logging.getLogger(__name__)


class TensorRTRunner:
    """TensorRT runner for model acceleration.

    This class provides functionality to run TensorRT engines from PyTorch models.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the TensorRT runner.

        Args:
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(*args, **kwargs)
        self._context = None
        self._io_tensors = None
        self._input_names = None
        self._output_names = None
        self._engine_info = None


@dataclass
class TensorRTBackendConfig(BackendConfig):
    """Configuration for TensorRT backend."""

    use_dynamo: bool = True
    workspace_size: int | None = None
    opset_version: int | None = None
    optimization_level: int | None = None
    compatibility_level: int | None = None
    timing_cache: Path | None = None
    profiles: list[TensorRTProfile] | None = None
    device: str = "cuda"
    quantization_config: ONNXAutoCastConfig | ONNXQuantizationConfig | TorchQuantizationConfig | None = None
    enable_tf32: bool = True
    use_cuda_graphs: bool = False

    def _default_describe_fields(self) -> list[str]:
        """Returns the default fields to describe."""
        return ["quantization_config"]

    @classmethod
    def from_dict(cls, state_dict: dict):
        """Convert dict to TensorRTBackendConfig."""
        return cls(**state_dict)


class TensorRTBackend(Backend, TensorRTRunner):
    """TensorRT backend for model acceleration.

    This class provides functionality to build and run TensorRT engines from PyTorch models.
    It handles the process of exporting models to ONNX and then converting them to TensorRT
    engines for optimized inference.
    """

    is_jit = False

    # State dictionary keys
    STATE_TYPE = "type"
    STATE_ENGINE_PATH = "engine_path"
    STATE_OUTPUT_OBJECT = "output_object"
    STATE_GRAPH_SPEC = "graph_spec"
    STATE_DEVICE = "device"
    STATE_QUANTIZATION_CONFIG = "quantization_config"
    STATE_CONFIG = "config"
    STATE_USE_CUDA_GRAPHS = "use_cuda_graphs"

    def __init__(
        self,
        config: TensorRTBackendConfig | None = None,
    ):
        """Initialize the TensorRT backend.

        Args:
            config: Configuration for TensorRT backend
        """
        super().__init__()

        # Create system monitor for tracking memory usage
        self._system_monitor = SystemMonitor()

        self._config = config or TensorRTBackendConfig()

        self._torch_model_info = None
        self._context = None
        self._io_tensors = None
        self._output_names = None
        self._input_names = None
        self._engine_info = None
        self._cuda_stream = None
        self._start_time = None
        self._end_time = None
        self._outputs = None

        # build variables
        self._engine_path = None
        self._output_object = None
        self._graph_spec = None

        # runtime variables
        self._output_allocator = None
        self._trt_runtime = None

        # CUDA graph variables
        self._cuda_graph = None
        self._last_input_shapes = None
        self._static_inputs = {}
        self._infer_cuda_graph = None

    def key(self) -> str:
        """Returns the key of the backend."""
        return f"{self.__class__.__name__}_{self._config.key()}"

    def describe(self) -> str:
        """Returns the description of the backend."""
        return f"{self.__class__.__name__}({self._config.describe()})"

    def _prepare_onnx_model_path(self, cache_dir: Path, suffix: str = "") -> Path:
        """Prepare the ONNX model path.

        Args:
            cache_dir (Path): The cache directory to store the ONNX model.
            suffix (str): The suffix of the ONNX model.

        Returns:
            Path: The ONNX model path
        """
        if suffix:
            suffix = f"_{suffix}"

        onnx_dir = cache_dir / f"onnx{suffix}"

        onnx_dir.mkdir(parents=True, exist_ok=True)
        return onnx_dir / f"model{ONNX_FILE_EXTENSION}"

    def _prepare_trt_engine_path(self, cache_dir: Path) -> Path:
        """Prepare the TensorRT engine path.

        Args:
            cache_dir (Path): The cache directory to store the TensorRT model.
        """
        engine_dir = cache_dir / "tensorrt"
        engine_dir.mkdir(parents=True, exist_ok=True)
        return engine_dir / f"model{TRT_ENGINE_FILE_EXTENSION}"

    def _build(self, module: nn.Module, graph_spec: GraphSpec, data: list[Sample], cache_dir: Path) -> Backend:
        """Build the TensorRT model.

        This method will export the module to ONNX and build a TensorRT engine from it using Polygraphy.

        Args:
            module (nn.Module): The module to build the TensorRT model for.
            name (str): The name of the model.
            graph_spec (GraphSpec): The graph spec of the model.
            data (list[Sample]): The data of the model.
            cache_dir (Path): The cache directory to store the TensorRT model.

        Returns:
            Backend: The backend with built TensorRT model.
        """
        self._graph_spec = graph_spec

        cuda_set_device(self._device)
        self._save_config(cache_dir)

        self._output_object = self._get_output_object(module=module, sample=data[0])

        if isinstance(self._config.quantization_config, TorchQuantizationConfig):
            self._build_modelopt_torch(module, graph_spec, data, cache_dir)
        elif isinstance(self._config.quantization_config, ONNXQuantizationConfig):
            self._build_modelopt_onnx(module, graph_spec, data, cache_dir)
        elif isinstance(self._config.quantization_config, ONNXAutoCastConfig):
            self._build_modelopt_onnx_autocast(module, graph_spec, data, cache_dir)
        else:
            self._build_standard(module, graph_spec, data, cache_dir)
        self._activate()

        return self

    def _build_modelopt_torch(self, module: nn.Module, graph_spec: GraphSpec, data: list[Sample], cache_dir: Path):
        """Build the TensorRT model  ModelOpt torch quantization.

        This method will quantize module using ModelOpt torch quantization, export the quantized model to ONNX and build a TensorRT engine from it using Polygraphy.

        Args:
            module (nn.Module): The module to build the TensorRT model for.
            name (str): The name of the model.
            graph_spec (GraphSpec): The graph spec of the model.
            data (list[Sample]): The data of the model.
            cache_dir (Path): The cache directory to store the TensorRT model.

        """
        logger.info("Using torch quantization with config: %s", self._config.quantization_config)
        logger.info("Using %s samples for quantization calibration", len(data))
        try:
            with self._system_monitor.system_stats_context(log_label="ModelOpt Torch quantization"):
                torch_quantizer = TorchQuantizer()
                module = torch_quantizer.quantize(
                    module=module,
                    sample=data[0],
                    config=self._config.quantization_config,
                )

            with self._system_monitor.system_stats_context(log_label="ONNX export"):
                self._onnx_path_quantized = self._prepare_onnx_model_path(cache_dir, suffix="ptq")

                self._onnx_exporter = ONNXExporter(
                    use_dynamo=self._config.use_dynamo,
                    opset_version=self._config.opset_version,
                    output_path=self._onnx_path_quantized,
                )
                self._onnx_exporter.export(
                    module=module,
                    sample=data[0],
                    graph_spec=graph_spec,
                    verbose=False,  # Disable verbose ONNX export to avoid excessive internal C++ logging
                )

                self._offload_torch_model_to_cpu(module)

            with self._system_monitor.system_stats_context(log_label="TensorRT engine build"):
                # Initialize TensorRT builder
                logger.info("Initializing TensorRT builder")
                min_shapes, opt_shapes, max_shapes = self._get_shapes(graph_spec=graph_spec)

                self._engine_path = self._prepare_trt_engine_path(cache_dir)

                self._trt_builder = TensorRTBuilder(
                    input_onnx_path=self._onnx_path_quantized,
                    output_path=self._engine_path,
                    workspace_size=self._config.workspace_size,
                    optimization_level=self._config.optimization_level,
                    compatibility_level=self._config.compatibility_level,
                    timing_cache=self._config.timing_cache,
                    profiles=self._config.profiles,
                    enable_tf32=self._config.enable_tf32,
                    min_shapes=min_shapes,
                    opt_shapes=opt_shapes,
                    max_shapes=max_shapes,
                )

                self._trt_builder.build()
        except Exception as e:
            # Using this except block to clean up any partial resources
            self._deactivate()
            raise e

    def _build_modelopt_onnx(self, module: nn.Module, graph_spec: GraphSpec, data: list[Sample], cache_dir: Path):
        """Build the TensorRT model ModelOpt ONNX quantization.

        This method will export the module to ONNX, quantize the exported model using ModelOpt ONNX quantization and build a TensorRT engine from it using Polygraphy.

        Args:
            module (nn.Module): The module to build the TensorRT model for.
            name (str): The name of the model.
            graph_spec (GraphSpec): The graph spec of the model.
            data (list[Sample]): The data of the model.
            cache_dir (Path): The cache directory to store the TensorRT model.
        """
        try:
            with self._system_monitor.system_stats_context(log_label="ONNX export"):
                logger.info("Initializing ONNX exporter")

                self._onnx_path = self._prepare_onnx_model_path(cache_dir)
                self._onnx_exporter = ONNXExporter(
                    use_dynamo=self._config.use_dynamo,
                    opset_version=self._config.opset_version,
                    output_path=self._onnx_path,
                )

                self._onnx_exporter.export(
                    module=module,
                    sample=data[0],
                    graph_spec=graph_spec,
                    verbose=False,  # Disable verbose ONNX export to avoid excessive internal C++ logging
                )

                self._offload_torch_model_to_cpu(module)

            with self._system_monitor.system_stats_context(log_label="ONNX quantization"):
                logger.info("Initializing ONNX quantizer")
                onnx_quantizer = ONNXQuantizer()

                self._onnx_path_quantized = self._prepare_onnx_model_path(cache_dir, suffix="ptq")

                # Quantize the ONNX model
                self._onnx_path_quantized = onnx_quantizer.quantize(
                    input_onnx_path=self._onnx_path,
                    output_path=self._onnx_path_quantized,
                    config=self._config.quantization_config,
                    samples=data,
                    graph_spec=graph_spec,
                )

            with self._system_monitor.system_stats_context(log_label="TensorRT engine build"):
                # Initialize TensorRT builder
                logger.info("Initializing TensorRT builder")
                min_shapes, opt_shapes, max_shapes = self._get_shapes(graph_spec=graph_spec)

                self._engine_path = self._prepare_trt_engine_path(cache_dir)

                self._trt_builder = TensorRTBuilder(
                    output_path=self._engine_path,
                    input_onnx_path=self._onnx_path_quantized,
                    workspace_size=self._config.workspace_size,
                    optimization_level=self._config.optimization_level,
                    compatibility_level=self._config.compatibility_level,
                    timing_cache=self._config.timing_cache,
                    profiles=self._config.profiles,
                    enable_tf32=self._config.enable_tf32,
                    min_shapes=min_shapes,
                    opt_shapes=opt_shapes,
                    max_shapes=max_shapes,
                )

                self._trt_builder.build()

        except Exception as e:
            # Use this except block to clean up any partial resources
            self._deactivate()
            raise e

    def _build_modelopt_onnx_autocast(
        self, module: nn.Module, graph_spec: GraphSpec, data: list[Sample], cache_dir: Path
    ):
        """Build the TensorRT model.

        This method will export the module to ONNX and build a TensorRT engine from it using Polygraphy.

        Args:
            module (nn.Module): The module to build the TensorRT model for.
            name (str): The name of the model.
            graph_spec (GraphSpec): The graph spec of the model.
            data (list[Sample]): The data of the model.
            cache_dir (Path): The cache directory to store the TensorRT model.
        """
        logger.info("Starting TensorRT backend building")
        try:
            with self._system_monitor.system_stats_context(log_label="ONNX export"):
                logger.info("Initializing ONNX exporter")

                self._onnx_path = self._prepare_onnx_model_path(cache_dir)
                self._onnx_exporter = ONNXExporter(
                    use_dynamo=self._config.use_dynamo,
                    opset_version=self._config.opset_version,
                    output_path=self._onnx_path,
                )

                self._onnx_exporter.export(
                    module=module,
                    sample=data[0],
                    graph_spec=graph_spec,
                    verbose=False,  # Disable verbose ONNX export to avoid excessive internal C++ logging
                )

                self._offload_torch_model_to_cpu(module)

            with self._system_monitor.system_stats_context(log_label="ONNX autocast"):
                logger.info("Initializing ONNX autocast")
                onnx_autocast = ONNXAutoCast()

                self._onnx_path_autocasted = self._prepare_onnx_model_path(cache_dir, suffix="autocast")

                self._onnx_path_autocasted = onnx_autocast.autocast(
                    input_onnx_path=self._onnx_path,
                    output_path=self._onnx_path_autocasted,
                    config=self._config.quantization_config,
                    samples=data,
                    graph_spec=graph_spec,
                )

            with self._system_monitor.system_stats_context(log_label="TensorRT engine build"):
                # Initialize TensorRT builder
                logger.info("Initializing TensorRT builder")
                min_shapes, opt_shapes, max_shapes = self._get_shapes(graph_spec=graph_spec)

                self._engine_path = self._prepare_trt_engine_path(cache_dir)

                self._trt_builder = TensorRTBuilder(
                    input_onnx_path=self._onnx_path_autocasted,
                    output_path=self._engine_path,
                    workspace_size=self._config.workspace_size,
                    optimization_level=self._config.optimization_level,
                    compatibility_level=self._config.compatibility_level,
                    timing_cache=self._config.timing_cache,
                    profiles=self._config.profiles,
                    enable_tf32=self._config.enable_tf32,
                    min_shapes=min_shapes,
                    opt_shapes=opt_shapes,
                    max_shapes=max_shapes,
                )

                self._trt_builder.build()
        except Exception as e:
            # Use this except block to clean up any partial resources
            self._deactivate()
            raise e

        logger.info(
            "TensorRT backend building finished successfully with engine path %s",
            self._engine_path,
        )

    def _build_standard(self, module: nn.Module, graph_spec: GraphSpec, data: list[Sample], cache_dir: Path):
        """Build the TensorRT model.

        This method will export the module to ONNX and build a TensorRT engine from it using Polygraphy.

        Args:
            module (nn.Module): The module to build the TensorRT model for.
            name (str): The name of the model.
            graph_spec (GraphSpec): The graph spec of the model.
            data (list[Sample]): The data of the model.
            cache_dir (Path): The cache directory to store the TensorRT model.
        """
        logger.info("Starting TensorRT backend building")
        try:
            with self._system_monitor.system_stats_context(log_label="ONNX export"):
                logger.info("Initializing ONNX exporter")

                self._onnx_path = self._prepare_onnx_model_path(cache_dir)
                self._onnx_exporter = ONNXExporter(
                    use_dynamo=self._config.use_dynamo,
                    opset_version=self._config.opset_version,
                    output_path=self._onnx_path,
                )

                self._onnx_exporter.export(
                    module=module,
                    sample=data[0],
                    graph_spec=graph_spec,
                    verbose=False,  # Disable verbose ONNX export to avoid excessive internal C++ logging
                )

                self._offload_torch_model_to_cpu(module)

            with self._system_monitor.system_stats_context(log_label="TensorRT engine build"):
                # Initialize TensorRT builder
                logger.info("Initializing TensorRT builder")
                min_shapes, opt_shapes, max_shapes = self._get_shapes(graph_spec=graph_spec)

                self._engine_path = self._prepare_trt_engine_path(cache_dir)

                self._trt_builder = TensorRTBuilder(
                    input_onnx_path=self._onnx_path,
                    output_path=self._engine_path,
                    workspace_size=self._config.workspace_size,
                    optimization_level=self._config.optimization_level,
                    compatibility_level=self._config.compatibility_level,
                    timing_cache=self._config.timing_cache,
                    profiles=self._config.profiles,
                    enable_tf32=self._config.enable_tf32,
                    min_shapes=min_shapes,
                    opt_shapes=opt_shapes,
                    max_shapes=max_shapes,
                )

                self._trt_builder.build()
        except Exception as e:
            # Use this except block to clean up any partial resources
            self._deactivate()
            raise e

        logger.info(
            "TensorRT backend building finished successfully with engine path %s",
            self._engine_path,
        )

    @nvtx.annotate(message="TensorRTBackend.infer", domain="AITune", color="green")
    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        """Infer using the TensorRT engine.

        Args:
            *args: Input tensors
            **kwargs: Named input tensors

        Returns:
            Model outputs
        """
        with torch.inference_mode():
            try:
                if self._context is None:
                    raise RuntimeError("Engine not loaded. Call build() first.")

                # Prepare inputs
                inputs = self._prepare_inputs(args, kwargs)
                if not inputs:
                    raise ValueError("No input tensors provided for inference")

                # Check if input shapes have changed (for CUDA graph invalidation)
                self._invalidate_cuda_graph(inputs)

                # Set input tensor shapes and addresses
                logger.debug("Setting input tensor shapes and addresses")
                self._set_input_tensors(inputs)

                # Run inference with timing
                logger.debug("Executing TensorRT inference")
                try:
                    # Only synchronize before timing for accuracy
                    with torch.cuda.stream(self._cuda_stream):
                        self._start_time.record(stream=self._cuda_stream)

                        # Using if instead overloading/replacing a function as simple if is faster
                        if self._config.use_cuda_graphs:
                            self._infer_cuda_graph()
                        else:
                            status = self._context.execute_async_v3(self._cuda_stream.cuda_stream)
                            if not status:
                                raise RuntimeError("TensorRT execution failed")

                    self._end_time.record(stream=self._cuda_stream)
                except Exception as e:
                    # Attempt to recover from error
                    torch.cuda.empty_cache()
                    raise e

                # Wait for inference to complete
                try:
                    logger.debug("Synchronizing CUDA stream")
                    self._cuda_stream.synchronize()

                    elapsed_time = self._start_time.elapsed_time(self._end_time)
                    logger.debug("Inference completed in %s ms", elapsed_time)
                except Exception as e:
                    torch.cuda.empty_cache()
                    elapsed_time = 0
                    raise e

                # Return a copy of the outputs with the correct format
                return self._prepare_outputs_for_return()

            except Exception as e:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                raise e

    def _activate(self):
        """Activate the TensorRT engine."""
        logger.debug("Activating TensorRT backend")

        cuda_set_device(self._device)

        self._trt_runtime = TensorRTRuntime()
        with self._system_monitor.system_stats_context(log_label="Loading engine"):
            engine_bytes = self._trt_runtime.load_engine(engine_path=self._engine_path)

        with self._system_monitor.system_stats_context(log_label="Creating execution context"):
            self._context, self._io_tensors, self._input_names, self._output_names, self._engine_info = (
                self._trt_runtime.create_execution_context(engine_bytes=engine_bytes)
            )

        # Initialize CUDA stream for inference
        logger.debug("Creating dedicated CUDA stream for inference")
        self._cuda_stream = torch.cuda.Stream()

        # Initializing CUDA graph for inference
        logger.debug("Creating CUDA graph for inference")
        if self._config.use_cuda_graphs:
            self._infer_cuda_graph = self._build_cuda_graph
            self._static_inputs = {}
            self._cuda_graph = None
        else:
            self._infer_cuda_graph = None

        # Initialize timing events
        logger.debug("Creating CUDA events for timing")
        self._start_time = torch.cuda.Event(enable_timing=True)
        self._end_time = torch.cuda.Event(enable_timing=True)

        # Initialize output allocator with engine info for proper dtype handling
        logger.debug("Setting up output allocator")
        self._output_allocator = TorchOutputAllocator(engine_info=self._engine_info)

        # Set output allocator for each output tensor individually
        for output_name in self._output_names:
            success = self._context.set_output_allocator(output_name, self._output_allocator)
            if not success:
                logger.error("Failed to set output allocator for tensor '%s'", output_name)
                raise RuntimeError(f"Failed to set output allocator for tensor '{output_name}'")
            logger.debug("Set output allocator for tensor '%s'", output_name)

    def _deactivate(self):
        """Deactivate the TensorRT engine."""
        logger.debug("Deactivating TensorRT backend")
        self._trt_runtime = None

        with self._system_monitor.system_stats_context(log_label="Deactivation"):
            try:
                with contextlib.ExitStack() as stack:
                    if self._context:
                        stack.enter_context(self._context)

                if self._output_allocator is not None:
                    self._output_allocator.clear()

                # Safely delete attributes if they exist
                for attr_name in [
                    "_io_tensors",
                    "_input_names",
                    "_output_names",
                    "_engine_info",
                    "_cuda_stream",
                    "_start_time",
                    "_end_time",
                    "_outputs",
                    "_context",
                    "_output_allocator",
                    "_cuda_graph",
                    "_last_input_shapes",
                    "_static_inputs",
                    "_infer_cuda_graph",
                ]:
                    if hasattr(self, attr_name):
                        delattr(self, attr_name)

            except Exception as e:
                logger.error("Failed to delete TensorRT objects: %s", e)

    def _deploy(self):
        """Deploys the backend.

        After deploying, the backend is ready to do inference. Backend cannot be deactivated anymore.
        """
        self._activate()

    def _get_output_object(self, module: nn.Module, sample: Sample) -> Any:
        """Get the output object from the module and sample.

        Args:
            module: PyTorch module
            sample: Sample input to use for model inference.

        Returns:
            The output object from the module.

        Note: to avoid case where a module returns a reference to the input argument, we make a deep copy of
        the output object.
        """
        module.to(self._device)
        args, kwargs = sample
        with torch.no_grad():
            output_object = module(*args, **kwargs)
        return copy.deepcopy(output_object)

    def _save_config(self, cache_dir: Path):
        """Store the backend configuration to a file."""
        config_path = cache_dir / "config.json"
        self._config.to_json(config_path)
        logger.info("Config saved to %s", config_path)

    def _offload_torch_model_to_cpu(self, model: "torch.nn.Module"):
        """Offload PyTorch model to meta device.

        Args:
            model: PyTorch model to offload.
        """
        logger.info("Offloading model to cpu device")
        model = model.to("cpu")

        gc.collect()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _prepare_outputs_for_return(self) -> Any:
        """Prepare the outputs for return.

        This method will prepare the outputs for return according to the original model's output structure.
        Tensors are retrieved from the output allocator with correct shapes.

        Returns:
            The outputs in the same format as the original model.
        """
        # Get outputs from the allocator (already properly shaped by TensorRT)
        outputs = self._output_allocator.outputs

        result = copy.deepcopy(self._output_object)  # make each results a unique copy of the original output object
        for locator, tensor_spec in self._graph_spec.output_spec.tensor_data:
            if tensor_spec.name in outputs:
                result = locator.set_value(result, outputs[tensor_spec.name].clone())
                del outputs[tensor_spec.name]
            else:
                logger.debug("Output: %s not found in outputs", tensor_spec.name)

        if len(outputs) > 0:
            logger.warning("Outputs not found in graph spec.")
            if isinstance(outputs, dict):
                for name, tensor in outputs.items():
                    logger.debug(" - %s not found in graph spec: %s", name, tensor.shape)
            elif isinstance(outputs, list):
                for tensor in outputs:
                    logger.debug(" - not found in graph spec: %s", tensor.shape)

        return result

    def _prepare_inputs(self, args, kwargs):
        """Prepare input tensors from args and kwargs.

        Args:
            args: Positional input tensors
            kwargs: Named input tensors

        Returns:
            dict: Dictionary mapping input names to tensors

        Raises:
            ValueError: If inputs are missing or incorrect
        """
        # Use input_names property directly from engine_info
        engine_input_names = set(self._engine_info.input_names)
        inputs = {}
        for locator, tensor_spec in self._graph_spec.input_spec.tensor_data:
            if tensor_spec.name in engine_input_names:
                if tensor_spec.name.startswith("args"):
                    inputs[tensor_spec.name] = locator.get_value(args)
                else:
                    inputs[tensor_spec.name] = locator.get_value(kwargs)
            else:
                logger.debug("Input: %s not found in inputs", tensor_spec.name)

        logger.debug("Prepared %s inputs for inference", len(inputs))
        logger.debug("Inputs names: %s", engine_input_names)
        return inputs

    def _set_input_tensors(self, inputs):
        """Set shapes and memory addresses for input tensors.

        Args:
            inputs: Dictionary mapping input names to tensors
        """
        for name, tensor in inputs.items():
            # Ensure tensor is contiguous and on CUDA
            if not tensor.is_contiguous():
                logger.debug("Input tensor %s is not contiguous, making it contiguous", name)
                tensor = tensor.contiguous()

            # Ensure tensor is on CUDA
            if not tensor.is_cuda:
                logger.debug("Input tensor %s is not on CUDA, moving it to CUDA", name)
                tensor = tensor.cuda()

            # Store the modified tensor back in inputs
            inputs[name] = tensor

            # Convert tensor shape to tuple for logging and TensorRT
            shape = tuple(tensor.shape)
            dtype = tensor.dtype
            logger.debug("Setting shape for input tensor %s: %s, dtype=%s", name, shape, dtype)

            # CUDA graphs always use the same memory address for the same input tensors
            if self._config.use_cuda_graphs:
                if name not in self._static_inputs:
                    # creating static memory for inputs
                    self._static_inputs[name] = torch.zeros_like(tensor)

                # copy the tensor to the static input
                static_tensor = self._static_inputs[name]
                static_tensor.copy_(tensor)

                tensor = static_tensor

                # TODO: can we have user provided static inputs?

            try:
                # Set input tensor shape directly
                self._context.set_input_shape(name, shape)
                logger.debug("Set input shape for %s successfully", name)
            except Exception as e:
                logger.error("Failed to set shape for %s: %s", name, e)
                raise e

            try:
                ptr = tensor.data_ptr()
                self._context.set_tensor_address(name, ptr)
                logger.debug("Set tensor address for %s successfully", name)
            except Exception as e:
                logger.error("Failed to set tensor address for %s: %s", name, e)
                raise e

    def _get_shapes(
        self, graph_spec: GraphSpec
    ) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
        """Get the shapes for the input tensors.

        Args:
            graph_spec: Input graph spec

        Returns:
            Tuple of dictionaries of minimum, optimal, and maximum input shapes
        """
        min_shapes = {}
        opt_shapes = {}
        max_shapes = {}
        for tensor_spec in graph_spec.input_spec.tensor_specs:
            min_shapes[tensor_spec.name] = tuple(tensor_spec.min_shape)
            opt_shapes[tensor_spec.name] = tuple(tensor_spec.max_shape)
            max_shapes[tensor_spec.name] = tuple(tensor_spec.max_shape)

        return min_shapes, opt_shapes, max_shapes

    def _build_cuda_graph(self):
        """Create a CUDA graph for inference."""
        logger.debug("Capturing CUDA graph for first time")
        # First execution to ensure everything is set up
        status = self._context.execute_async_v3(self._cuda_stream.cuda_stream)
        if not status:
            raise RuntimeError("TensorRT execution failed during CUDA graph setup")

        # Begin CUDA graph capture
        self._cuda_graph = torch.cuda.CUDAGraph()

        # See https://docs.pytorch.org/docs/2.4/generated/torch.cuda.graph.html for more args
        with torch.cuda.graph(self._cuda_graph, stream=self._cuda_stream):
            # Execute inference within the capture
            status = self._context.execute_async_v3(self._cuda_stream.cuda_stream)
            if not status:
                raise RuntimeError("TensorRT execution failed during CUDA graph capture")

        logger.debug("CUDA graph captured and instantiated successfully")
        self._infer_cuda_graph = self._execute_cuda_graph

        # Execute the CUDA graph
        self._infer_cuda_graph()

    def _execute_cuda_graph(self):
        """Execute inference using CUDA graphs for optimized performance.

        This method implements the CUDA graph capture and launch pattern:
        1. If no CUDA graph exists, capture one by running inference twice
        2. If CUDA graph exists, launch the captured graph
        """
        logger.debug("Launching CUDA graph")
        # Launch the captured CUDA graph
        self._cuda_graph.replay()

    def _invalidate_cuda_graph(self, inputs: dict[str, torch.Tensor]):
        """Setup the inputs for the CUDA graph.

        This should be called when input shapes change or when the graph needs to be rebuilt.
        """
        if self._config.use_cuda_graphs:
            current_input_shapes = {name: tensor.shape for name, tensor in inputs.items()}
            if self._last_input_shapes != current_input_shapes:
                logger.debug("Input shapes changed, invalidating CUDA graph")
                self._cuda_graph = None
                self._infer_cuda_graph = self._build_cuda_graph
                self._last_input_shapes = current_input_shapes
                self._static_inputs = {}

    def to_dict(self):
        """Returns the state_dict of the backend.

        Returns:
            dict: Dictionary containing the backend state
        """
        return {
            self.STATE_TYPE: self.__class__.__name__,
            self.STATE_ENGINE_PATH: self._engine_path,
            self.STATE_OUTPUT_OBJECT: self._output_object,
            self.STATE_GRAPH_SPEC: self._graph_spec.to_dict(),
            self.STATE_DEVICE: self._device,
            self.STATE_QUANTIZATION_CONFIG: self._config.quantization_config,
            self.STATE_CONFIG: self._config.to_dict(),
            self.STATE_USE_CUDA_GRAPHS: self._config.use_cuda_graphs,
        }

    @classmethod
    def from_dict(cls, module: torch.nn.Module, state_dict: dict):
        """Creates a backend from a state_dict.

        Args:
            module: The PyTorch module
            state_dict: Dictionary containing the backend state

        Returns:
            TensorRTBackend: The reconstructed backend
        """
        backend = cls()
        backend._engine_path = state_dict[cls.STATE_ENGINE_PATH]
        backend._graph_spec = GraphSpec.from_dict(state_dict[cls.STATE_GRAPH_SPEC])
        backend._device = state_dict[cls.STATE_DEVICE]
        backend.state = BackendState.CHECKPOINT_LOADED

        # Reconstruct config with quantization settings
        backend._config = TensorRTBackendConfig.from_dict(state_dict[cls.STATE_CONFIG])

        backend._output_object = state_dict[cls.STATE_OUTPUT_OBJECT]
        # Ensure CUDA graphs are disabled when loading from checkpoint
        # CUDA graphs cannot be serialized and must be re-captured
        if backend._config.use_cuda_graphs:
            logger.info("CUDA graphs were enabled in saved state, but will be re-captured on first inference")
            # CUDA graph state will be None initially, triggering re-capture

        return backend
