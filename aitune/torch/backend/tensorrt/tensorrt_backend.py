# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TensorRT backend."""

import contextlib
import copy
import json
import logging
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, cast

import torch
import torch.nn as nn
from polygraphy.backend.trt import Profile
from polygraphy.logger import G_LOGGER

from aitune.exceptions import AITuneUserInputError
from aitune.records import BoundedTensorSpec, DeploymentArtifact, ModelFiles, RuntimeConfig, TensorSample
from aitune.torch.artifact import artifact_input_sample, bounded_tensor_specs
from aitune.torch.backend.backend import (
    Backend,
    BackendBuildStep,
    BackendConfig,
    BackendState,
    BuildMode,
    ExecutionMode,
    ModuleFormat,
)
from aitune.torch.backend.tensorrt.cuda_graphs import CudaGraphCachePolicy, TensorRTCudaGraphCache
from aitune.torch.backend.tensorrt.onnx_autocast import ONNXAutoCast, ONNXAutoCastConfig
from aitune.torch.backend.tensorrt.onnx_quantization import ONNXQuantizationConfig, ONNXQuantizer
from aitune.torch.backend.tensorrt.tensorrt_builder import TensorRTBuilder
from aitune.torch.backend.tensorrt.tensorrt_profile import TensorRTProfile
from aitune.torch.backend.tensorrt.tensorrt_runtime import TensorRTRuntime
from aitune.torch.backend.tensorrt.torch_output_allocator import TorchOutputAllocator
from aitune.torch.backend.tensorrt.torch_quantization import TorchQuantizationConfig, TorchQuantizer
from aitune.torch.checkpoint.artifact import ArtifactPath
from aitune.torch.config import config as global_config
from aitune.torch.distributed import distributed_output_path
from aitune.torch.libs.onnx.onnx_exporter import ONNXExporter
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.onnx_module import OnnxModule
from aitune.torch.module.sample_store import Sample, SampleStore
from aitune.torch.utils.cuda_utils import set_device as cuda_set_device
from aitune.torch.utils.module import offload
from aitune.torch.utils.tensor import format_tensor_name
from aitune.utils.monitoring import annotate

G_LOGGER.use_python_logging_system = True

# File extension constants
TRT_ENGINE_FILE_EXTENSION = ".plan"
ONNX_FILE_EXTENSION = ".onnx"


# Setup logger
logger = logging.getLogger(__name__)


class TensorRTBuildStep(BackendBuildStep):
    """Identifiers for discrete sub-steps of a TensorRT backend build."""

    ONNX_EXPORT = "ONNX export"
    TENSORRT_ENGINE_BUILD = "TensorRT engine build"
    MODELOPT_TORCH_QUANTIZATION = "ModelOpt Torch quantization"
    MODELOPT_ONNX_QUANTIZATION = "ModelOpt ONNX quantization"
    ONNX_AUTOCAST = "ONNX autocast"


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


class ProfileMode(Enum):
    """Mode how TRT optimization profiles will be generated for TensorRT engine.

    Attributes:
        SINGLE: auto-generate single profile from graph spec, default mode.
        SAMPLES_USED: auto-generated multiple profiles from shapes of samples used for tuning.
    """

    SINGLE = "single"
    SAMPLES_USED = "samples_used"


@dataclass
class TensorRTBackendConfig(BackendConfig):
    """Configuration for TensorRT backend.

    Attributes:
        use_dynamo: Whether to use torch.dynamo for export.
        workspace_size: The workspace size for the TensorRT engine.
        opset_version: The ONNX opset version to use for export.
        optimization_level: The optimization level for the TensorRT engine.
        compatibility_level: The compatibility level for the TensorRT engine.
        timing_cache: The path to the timing cache for the TensorRT engine.
        profiles: How TensorRT optimization profiles are generated.
            - SINGLE: auto-generate a single profile from the graph spec (default).
            - SAMPLES_USED: auto-generate multiple profiles from shapes of samples used for tuning.
            - list[TensorRTProfile]: use user-provided profiles directly.
        device: The device to use for the TensorRT engine.
        quantization_config: The quantization configuration for the TensorRT engine.
        enable_tf32: Whether to enable TF32 hardware acceleration.
        use_cuda_graphs: Cache CUDA graphs for static profiles (enabled by default), falling back on capture failure.
        max_cuda_graphs: Maximum cached CUDA graphs per backend.
        cuda_graph_cache_policy: Aged LFU admission and eviction by default, or unconditional LRU admission.
    """

    use_dynamo: bool = True
    workspace_size: int | None = None
    opset_version: int | None = None
    optimization_level: int | None = None
    compatibility_level: int | None = None
    timing_cache: Path | None = None
    profiles: ProfileMode | list[TensorRTProfile] = ProfileMode.SINGLE
    device: str = "cuda"
    quantization_config: ONNXAutoCastConfig | ONNXQuantizationConfig | TorchQuantizationConfig | None = None
    enable_tf32: bool = True
    use_cuda_graphs: bool = True
    max_cuda_graphs: int = 8
    cuda_graph_cache_policy: CudaGraphCachePolicy = "lfu"

    def __post_init__(self):
        """Validate the graph cache capacity and policy."""
        if (
            isinstance(self.max_cuda_graphs, bool)
            or not isinstance(self.max_cuda_graphs, int)
            or self.max_cuda_graphs < 1
        ):
            raise ValueError("max_cuda_graphs must be a positive integer")
        if self.cuda_graph_cache_policy not in ("lru", "lfu"):
            raise ValueError("cuda_graph_cache_policy must be 'lru' or 'lfu'")

    @classmethod
    def from_dict(cls, data: dict) -> "TensorRTBackendConfig":
        """Initialise config from a plain dict (e.g. parsed from YAML).

        ``profiles`` may be passed as a string (``ProfileMode`` value) or a
        list of profile dicts and will be reconstructed automatically.
        ``quantization_config`` may be passed as a dict with a ``_type`` key
        (produced by ``to_dict()``) and will be reconstructed automatically.
        """
        data = dict(data)
        if "profiles" in data:
            data["profiles"] = cls.profiles_from_dict(data["profiles"])
        if isinstance(data.get("quantization_config"), dict):
            data["quantization_config"] = cls.quantization_config_from_dict(data["quantization_config"])
        return cls(**data)

    @classmethod
    def profiles_from_dict(cls, data: str | list[dict]) -> ProfileMode | list[TensorRTProfile]:
        """Convert dict to list of TensorRTProfile."""
        if isinstance(data, list):
            return [TensorRTProfile.from_dict(profile) for profile in data]
        return ProfileMode(data)

    @classmethod
    def quantization_config_from_dict(
        cls, data: dict
    ) -> ONNXAutoCastConfig | ONNXQuantizationConfig | TorchQuantizationConfig:
        """Reconstruct a quantization config from a dict produced by ``to_dict()``.

        The dict must contain a ``_type`` key with the class name.
        """
        _type_map = {
            "ONNXAutoCastConfig": ONNXAutoCastConfig,
            "ONNXQuantizationConfig": ONNXQuantizationConfig,
            "TorchQuantizationConfig": TorchQuantizationConfig,
        }
        data = dict(data)
        type_name = data.pop("_type", None)
        if type_name not in _type_map:
            raise ValueError(f"Unknown quantization_config type: {type_name!r}. Expected one of {list(_type_map)}")
        return _type_map[type_name].from_dict(data)

    def to_dict(self) -> dict:
        """Convert TensorRTBackendConfig to dictionary."""
        state_dict = asdict(self)
        if isinstance(self.profiles, list):
            state_dict["profiles"] = [
                TensorRTProfile.profile_to_dict(profile.profile) for profile in state_dict["profiles"]
            ]
        else:
            state_dict["profiles"] = self.profiles.value
        if self.quantization_config is not None:
            state_dict["quantization_config"] = {
                "_type": type(self.quantization_config).__name__,
                **asdict(self.quantization_config),
            }
        return state_dict


class TensorRTBackend(Backend, TensorRTRunner):
    """TensorRT backend for model acceleration.

    This class builds and runs TensorRT engines from PyTorch models or existing OnnxModule graphs.
    Torch models are exported to ONNX; OnnxModule uses its source path directly.
    """

    _build_mode = BuildMode.AHEAD_OF_TIME
    _supported_modules = frozenset({ModuleFormat.TORCH, ModuleFormat.ONNX})
    _execution_modes = frozenset({ExecutionMode.SINGLE_GPU})

    # State dictionary keys
    STATE_TYPE = "type"
    STATE_ENGINE_PATH = "engine_path"
    STATE_TRT_OPTIMIZATION_PROFILES_PATH = "trt_optimization_profiles_path"
    STATE_OUTPUT_OBJECT = "output_object"
    STATE_GRAPH_SPEC = "graph_spec"
    STATE_DEVICE = "device"
    STATE_QUANTIZATION_CONFIG = "quantization_config"
    STATE_CONFIG = "config"
    STATE_USE_CUDA_GRAPHS = "use_cuda_graphs"
    STATE_SAMPLES = "samples"

    # Supported devices
    _devices: ClassVar[list[str]] = ["cuda"]

    def __init__(
        self,
        config: TensorRTBackendConfig | None = None,
    ):
        """Initialize the TensorRT backend.

        Args:
            config: Configuration for TensorRT backend
        """
        super().__init__()

        self._config = config or TensorRTBackendConfig()

        if self._config.profiles == ProfileMode.SAMPLES_USED and global_config.max_num_samples_stored <= 1:
            raise ValueError(
                """aitune.torch.config.max_num_samples_stored is set to 1, change it to number of samples to use for profile generation.
                Example:
                from aitune.torch.config import config as global_config
                global_config.max_num_samples_stored = <number of samples>
                """
            )

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
        self._engine_artifact: ArtifactPath | None = None
        self._trt_optimization_profiles_artifact: ArtifactPath | None = None
        self._output_object = None
        self._graph_spec = None
        self._samples: Sequence[Sample] | None = None

        # runtime variables
        self._output_allocator = None
        self._trt_runtime = None
        self._trt_optimization_profiles: list[Profile] = []

        self._cuda_graphs = TensorRTCudaGraphCache()
        self._base_context = None
        self._base_output_allocator = None

    def key(self) -> str:
        """Returns the key of the backend."""
        return f"{self.__class__.__name__}_{self._config.key()}"

    def describe(self) -> str:
        """Returns the description of the backend."""
        return f"{self.__class__.__name__}({self._config.describe()})"

    def artifact(self) -> DeploymentArtifact:
        """Create the deployment record on request, including after deactivation."""
        try:
            return self._create_artifact()
        except (OSError, RuntimeError, ValueError) as error:
            raise RuntimeError(f"TensorRT artifact is not available: {error}") from error

    def _timing_cache_path(self) -> Path | None:
        """Return a timing cache path that is safe for this process."""
        if self._config.timing_cache is None:
            return None

        timing_cache = Path(self._config.timing_cache)
        resolved_path = distributed_output_path(timing_cache)
        if resolved_path != timing_cache:
            logger.info("Using rank-isolated TensorRT timing cache: %s", resolved_path)
        return resolved_path

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

    def _prepare_trt_optimization_profiles_path(self, cache_dir: Path) -> Path:
        """Prepare the TensorRT optimization profiles path.

        Args:
            cache_dir (Path): The cache directory to store the TensorRT optimization profiles.
        """
        optimization_profiles_dir = cache_dir / "tensorrt"
        optimization_profiles_dir.mkdir(parents=True, exist_ok=True)
        return optimization_profiles_dir / "model_optimization_profiles.json"

    def _build(self, module: nn.Module, graph_spec: GraphSpec, samples: SampleStore, cache_dir: Path) -> Backend:
        """Build the TensorRT model.

        This method will export the module to ONNX and build a TensorRT engine from it using Polygraphy.

        Args:
            module (nn.Module): The module to build the TensorRT model for.
            name (str): The name of the model.
            graph_spec (GraphSpec): The graph spec of the model.
            samples: Recorded samples for the model.
            cache_dir (Path): The cache directory to store the TensorRT model.

        Returns:
            Backend: The backend with built TensorRT model.
        """
        logger.info("Starting TensorRT backend building")
        self._graph_spec = graph_spec
        self._samples = samples

        cuda_set_device(self._device)
        if isinstance(module, OnnxModule):
            module.deactivate()
            if isinstance(self._config.quantization_config, TorchQuantizationConfig):
                raise AITuneUserInputError(
                    "Torch quantization requires a Torch module; use ONNX quantization for OnnxModule."
                )
        self._output_object = (
            {spec.name: None for _, spec in graph_spec.output_spec.tensor_data}
            if isinstance(module, OnnxModule)
            else self._get_output_object(module=module, sample=samples[0])
        )

        if isinstance(self._config.quantization_config, TorchQuantizationConfig):
            engine_path = self._build_modelopt_torch(module, graph_spec, samples, cache_dir)
        elif isinstance(self._config.quantization_config, ONNXQuantizationConfig):
            engine_path = self._build_modelopt_onnx(module, graph_spec, samples, cache_dir)
        elif isinstance(self._config.quantization_config, ONNXAutoCastConfig):
            engine_path = self._build_modelopt_onnx_autocast(module, graph_spec, samples, cache_dir)
        else:
            engine_path = self._build_standard(module, graph_spec, samples, cache_dir)
        self._engine_artifact = ArtifactPath.from_existing(engine_path, root=cache_dir)

        # Save optimization profiles after building the engine
        profiles_path = self._save_trt_optimization_profiles(self._trt_optimization_profiles, cache_dir)
        self._trt_optimization_profiles_artifact = ArtifactPath.from_existing(
            profiles_path,
            root=cache_dir,
        )

        logger.info("TensorRT backend building finished successfully with engine path %s", self._engine_artifact)
        self._activate()

        return self

    def _build_modelopt_torch(
        self, module: nn.Module, graph_spec: GraphSpec, samples: Sequence[Sample], cache_dir: Path
    ) -> Path:
        """Build the TensorRT model  ModelOpt torch quantization.

        This method will quantize module using ModelOpt torch quantization, export the quantized model to ONNX and build a TensorRT engine from it using Polygraphy.

        Args:
            module (nn.Module): The module to build the TensorRT model for.
            name (str): The name of the model.
            graph_spec (GraphSpec): The graph spec of the model.
            samples: Recorded samples for the model.
            cache_dir (Path): The cache directory to store the TensorRT model.

        """
        try:
            step = TensorRTBuildStep.MODELOPT_TORCH_QUANTIZATION
            with annotate(step.annotation), self._track_build_step(step):
                torch_quantizer = TorchQuantizer()
                module = torch_quantizer.quantize(
                    module=module,
                    sample=samples[0],
                    config=self._config.quantization_config,
                )

            onnx_path_quantized = self._export_onnx(module, graph_spec, samples[0], cache_dir, suffix="ptq")

            with annotate("build: Offloading model to cpu device"):
                offload(module, device="cpu")

            step = TensorRTBuildStep.TENSORRT_ENGINE_BUILD
            with annotate(step.annotation), self._track_build_step(step) as result:
                # Initialize TensorRT builder
                logger.info("Initializing TensorRT builder")
                engine_path = self._prepare_trt_engine_path(cache_dir)
                self._trt_optimization_profiles = self.get_profiles(graph_spec=graph_spec, samples=samples)

                trt_builder = TensorRTBuilder(
                    input_onnx_path=onnx_path_quantized,
                    output_path=engine_path,
                    workspace_size=self._config.workspace_size,
                    optimization_level=self._config.optimization_level,
                    compatibility_level=self._config.compatibility_level,
                    timing_cache=self._timing_cache_path(),
                    profiles=self._trt_optimization_profiles,
                    enable_tf32=self._config.enable_tf32,
                )

                trt_builder.build()
                result["engine_size_bytes"] = engine_path.stat().st_size
                result["num_optimization_profiles"] = len(self._trt_optimization_profiles)

            return engine_path
        except Exception as e:
            # Use this except block to clean up any partial resources
            self._deactivate()
            raise e

    def _build_modelopt_onnx(
        self, module: nn.Module, graph_spec: GraphSpec, samples: Sequence[Sample], cache_dir: Path
    ) -> Path:
        """Build the TensorRT model ModelOpt ONNX quantization.

        This method will export the module to ONNX, quantize the exported model using ModelOpt ONNX quantization and build a TensorRT engine from it using Polygraphy.

        Args:
            module (nn.Module): The module to build the TensorRT model for.
            name (str): The name of the model.
            graph_spec (GraphSpec): The graph spec of the model.
            samples: Recorded samples for the model.
            cache_dir (Path): The cache directory to store the TensorRT model.
        """
        try:
            onnx_path = self._export_onnx(module, graph_spec, samples[0], cache_dir)

            with annotate("build: Offloading model to cpu device"):
                offload(module, device="cpu")

            step = TensorRTBuildStep.MODELOPT_ONNX_QUANTIZATION
            with annotate(step.annotation), self._track_build_step(step) as result:
                logger.info("Initializing ONNX quantizer")
                onnx_quantizer = ONNXQuantizer()

                onnx_path_quantized = self._prepare_onnx_model_path(cache_dir, suffix="ptq")

                # Quantize the ONNX model
                onnx_path_quantized = onnx_quantizer.quantize(
                    input_onnx_path=onnx_path,
                    output_path=onnx_path_quantized,
                    config=self._config.quantization_config,
                    samples=samples,
                    graph_spec=graph_spec,
                )
                result["onnx_size_bytes"] = onnx_path_quantized.stat().st_size

            step = TensorRTBuildStep.TENSORRT_ENGINE_BUILD
            with annotate(step.annotation), self._track_build_step(step) as result:
                # Initialize TensorRT builder
                logger.info("Initializing TensorRT builder")
                engine_path = self._prepare_trt_engine_path(cache_dir)
                self._trt_optimization_profiles = self.get_profiles(graph_spec=graph_spec, samples=samples)

                trt_builder = TensorRTBuilder(
                    input_onnx_path=onnx_path_quantized,
                    output_path=engine_path,
                    workspace_size=self._config.workspace_size,
                    optimization_level=self._config.optimization_level,
                    compatibility_level=self._config.compatibility_level,
                    timing_cache=self._timing_cache_path(),
                    profiles=self._trt_optimization_profiles,
                    enable_tf32=self._config.enable_tf32,
                )

                trt_builder.build()
                result["engine_size_bytes"] = engine_path.stat().st_size
                result["num_optimization_profiles"] = len(self._trt_optimization_profiles)

            return engine_path
        except Exception as e:
            # Use this except block to clean up any partial resources
            self._deactivate()
            raise e

    def _build_modelopt_onnx_autocast(
        self, module: nn.Module, graph_spec: GraphSpec, samples: Sequence[Sample], cache_dir: Path
    ) -> Path:
        """Build the TensorRT model.

        This method will export the module to ONNX and build a TensorRT engine from it using Polygraphy.

        Args:
            module (nn.Module): The module to build the TensorRT model for.
            name (str): The name of the model.
            graph_spec (GraphSpec): The graph spec of the model.
            samples: Recorded samples for the model.
            cache_dir (Path): The cache directory to store the TensorRT model.
        """
        try:
            onnx_path = self._export_onnx(module, graph_spec, samples[0], cache_dir)

            with annotate("build: Offloading model to cpu device"):
                offload(module, device="cpu")

            step = TensorRTBuildStep.ONNX_AUTOCAST
            with annotate(step.annotation), self._track_build_step(step) as result:
                logger.info("Initializing ONNX autocast")
                onnx_autocast = ONNXAutoCast()

                onnx_path_autocasted = self._prepare_onnx_model_path(cache_dir, suffix="autocast")

                onnx_path_autocasted = onnx_autocast.autocast(
                    input_onnx_path=onnx_path,
                    output_path=onnx_path_autocasted,
                    config=self._config.quantization_config,
                    samples=samples,
                    graph_spec=graph_spec,
                )
                result["onnx_size_bytes"] = onnx_path_autocasted.stat().st_size

            step = TensorRTBuildStep.TENSORRT_ENGINE_BUILD
            with annotate(step.annotation), self._track_build_step(step) as result:
                # Initialize TensorRT builder
                logger.info("Initializing TensorRT builder")
                engine_path = self._prepare_trt_engine_path(cache_dir)
                self._trt_optimization_profiles = self.get_profiles(graph_spec=graph_spec, samples=samples)

                trt_builder = TensorRTBuilder(
                    input_onnx_path=onnx_path_autocasted,
                    output_path=engine_path,
                    workspace_size=self._config.workspace_size,
                    optimization_level=self._config.optimization_level,
                    compatibility_level=self._config.compatibility_level,
                    timing_cache=self._timing_cache_path(),
                    profiles=self._trt_optimization_profiles,
                    enable_tf32=self._config.enable_tf32,
                )

                trt_builder.build()
                result["engine_size_bytes"] = engine_path.stat().st_size
                result["num_optimization_profiles"] = len(self._trt_optimization_profiles)

            return engine_path
        except Exception as e:
            # Use this except block to clean up any partial resources
            self._deactivate()
            raise e

    def _export_onnx(
        self, module: nn.Module, graph_spec: GraphSpec, sample: Sample, cache_dir: Path, suffix: str = ""
    ) -> Path:
        """Export Torch or copy an ONNX graph and its external weights into the backend cache."""
        step = TensorRTBuildStep.ONNX_EXPORT
        with annotate(step.annotation), self._track_build_step(step) as result:
            path = self._prepare_onnx_model_path(cache_dir, suffix)
            if isinstance(module, OnnxModule):
                path, size = (Path(module.path), module.path.stat().st_size)
            else:
                exporter = ONNXExporter(
                    use_dynamo=self._config.use_dynamo,
                    opset_version=self._config.opset_version,
                    output_path=path,
                )
                exporter.export(module=module, sample=sample, graph_spec=graph_spec)
                size = path.stat().st_size

            result["onnx_size_bytes"] = size

        return path

    def _build_standard(
        self, module: nn.Module, graph_spec: GraphSpec, samples: Sequence[Sample], cache_dir: Path
    ) -> Path:
        """Build the TensorRT model.

        This method will export the module to ONNX and build a TensorRT engine from it using Polygraphy.

        Args:
            module (nn.Module): The module to build the TensorRT model for.
            name (str): The name of the model.
            graph_spec (GraphSpec): The graph spec of the model.
            samples: Recorded samples for the model.
            cache_dir (Path): The cache directory to store the TensorRT model.
        """
        try:
            onnx_path = self._export_onnx(module, graph_spec, samples[0], cache_dir)

            with annotate("build: Offloading model to cpu device"):
                offload(module, device="cpu")

            step = TensorRTBuildStep.TENSORRT_ENGINE_BUILD
            with annotate(step.annotation), self._track_build_step(step) as result:
                # Initialize TensorRT builder
                logger.info("Initializing TensorRT builder")
                engine_path = self._prepare_trt_engine_path(cache_dir)
                self._trt_optimization_profiles = self.get_profiles(graph_spec=graph_spec, samples=samples)

                trt_builder = TensorRTBuilder(
                    input_onnx_path=onnx_path,
                    output_path=engine_path,
                    workspace_size=self._config.workspace_size,
                    optimization_level=self._config.optimization_level,
                    compatibility_level=self._config.compatibility_level,
                    timing_cache=self._timing_cache_path(),
                    profiles=self._trt_optimization_profiles,
                    enable_tf32=self._config.enable_tf32,
                )

                trt_builder.build()
                result["engine_size_bytes"] = engine_path.stat().st_size
                result["num_optimization_profiles"] = len(self._trt_optimization_profiles)

            return engine_path
        except Exception as e:
            # Use this except block to clean up any partial resources
            self._deactivate()
            raise e

    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        """Infer using the TensorRT engine.

        Args:
            *args: Input tensors
            **kwargs: Named input tensors

        Returns:
            Model outputs
        """
        with torch.no_grad():
            try:
                if self._context is None:
                    raise RuntimeError("Engine not loaded. Call build() first.")

                # Prepare inputs
                inputs = self._prepare_inputs(args, kwargs)
                if not inputs:
                    raise ValueError("No input tensors provided for inference")

                self._set_optimization_profiles(inputs)

                # Set input tensor shapes and addresses
                logger.debug("Setting input tensor shapes and addresses")
                self._set_input_tensors(inputs)

                # Run inference with timing
                logger.debug("Executing TensorRT inference")
                with torch.cuda.stream(self._cuda_stream):
                    self._start_time.record(stream=self._cuda_stream)
                    if self._use_cuda_graphs:
                        self._cuda_graphs.execute(self._cuda_stream)
                    else:
                        self._execute_engine()
                    self._end_time.record(stream=self._cuda_stream)

                # Wait for inference to complete
                self._cuda_stream.synchronize()
                if self._cuda_graphs.capture_failed:
                    self._cuda_graphs.clear()

                elapsed_time = self._start_time.elapsed_time(self._end_time)
                logger.debug("Inference completed in %s ms", elapsed_time)

                # Return a copy of the outputs with the correct format
                return self._prepare_outputs_for_return()

            except Exception:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                raise

    def _activate(self):
        """Activate the TensorRT engine."""
        logger.debug("Activating TensorRT backend")
        engine_artifact = cast(ArtifactPath, self._engine_artifact)
        optimization_profiles_artifact = cast(ArtifactPath, self._trt_optimization_profiles_artifact)

        cuda_set_device(self._device)

        self._trt_runtime = TensorRTRuntime()
        engine_bytes = self._trt_runtime.load_engine(engine_path=engine_artifact.path)

        self._context, self._io_tensors, self._input_names, self._output_names, self._engine_info = (
            self._trt_runtime.create_execution_context(engine_bytes=engine_bytes)
        )
        self._base_context = self._context

        # Initialize CUDA stream for inference
        logger.debug("Creating dedicated CUDA stream for inference")
        self._cuda_stream = torch.cuda.Stream()

        # Initialize timing events
        logger.debug("Creating CUDA events for timing")
        self._start_time = torch.cuda.Event(enable_timing=True)
        self._end_time = torch.cuda.Event(enable_timing=True)

        # Initialize output allocator with engine info for proper dtype handling
        logger.debug("Setting up output allocator")
        self._output_allocator = self._create_output_allocator(self._context)
        self._base_output_allocator = self._output_allocator

        # Load optimization profiles and classify graph eligibility from all inputs.
        self._trt_optimization_profiles = self._load_trt_optimization_profiles(optimization_profiles_artifact.path)
        self._cuda_graphs.configure(
            self._base_context.engine,
            self._io_tensors,
            self._input_names,
            self._trt_optimization_profiles,
            max_graphs=self._config.max_cuda_graphs,
            policy=self._config.cuda_graph_cache_policy,
        )

    def _create_output_allocator(self, context):
        """Attach an independent output allocator to an execution context."""
        allocator = TorchOutputAllocator(engine_info=self._engine_info)

        # Set output allocator for each output tensor individually
        for output_name in self._output_names:
            success = context.set_output_allocator(output_name, allocator)
            if not success:
                logger.error("Failed to set output allocator for tensor '%s'", output_name)
                raise RuntimeError(f"Failed to set output allocator for tensor '{output_name}'")
            logger.debug("Set output allocator for tensor '%s'", output_name)

        return allocator

    def _create_artifact(self) -> DeploymentArtifact:
        """Create an artifact from the saved engine interface and recorded bounds."""
        if (
            self._engine_artifact is None
            or self._graph_spec is None
            or self._input_names is None
            or self._output_names is None
        ):
            raise RuntimeError("TensorRT artifact requires a built or deployed backend with a recorded interface")

        profiles = self._artifact_profiles()
        inputs = bounded_tensor_specs(self._graph_spec, "input", recorded_names=self._input_names)
        self._validate_artifact_profile_input_bounds(inputs, profiles)
        inputs = tuple(
            replace(
                input_spec,
                min_shape=tuple(
                    min(profile[input_spec.name]["min_shape"][axis] for profile in profiles)
                    for axis in range(len(input_spec.min_shape))
                ),
                max_shape=tuple(
                    max(profile[input_spec.name]["max_shape"][axis] for profile in profiles)
                    for axis in range(len(input_spec.max_shape))
                ),
            )
            for input_spec in inputs
        )
        engine_path = self._engine_artifact.path
        return DeploymentArtifact(
            model=ModelFiles(
                format="tensorrt_plan",
                path=engine_path,
                metadata={"optimization_profiles": profiles, "optimization_profile_count": len(profiles)},
            ),
            inputs=inputs,
            outputs=bounded_tensor_specs(self._graph_spec, "output", recorded_names=self._output_names),
            runtime=RuntimeConfig(
                name="tensorrt",
                options={
                    "use_cuda_graphs": self._config.use_cuda_graphs,
                    "max_cuda_graphs": self._config.max_cuda_graphs,
                    "cuda_graph_cache_policy": self._config.cuda_graph_cache_policy,
                },
            ),
            sample_inputs=self._artifact_sample_inputs(),
        )

    def _artifact_sample_inputs(self) -> tuple[TensorSample, ...]:
        """Derive portable values from the checkpointed sample and GraphSpec."""
        if self._samples is None:
            return ()
        try:
            return artifact_input_sample(cast(GraphSpec, self._graph_spec), self._samples[0])
        except Exception as error:
            logger.info("Perf Analyzer will use synthetic inputs: %s", error)
            return ()

    @staticmethod
    def _validate_artifact_profile_input_bounds(
        inputs: tuple[BoundedTensorSpec, ...],
        profiles: tuple[dict[str, dict[str, tuple[int, ...]]], ...],
    ) -> None:
        """Require TensorRT input profiles to stay within recorded input bounds."""
        for profile_index, profile in enumerate(profiles):
            for input_spec in inputs:
                minimum = profile[input_spec.name]["min_shape"]
                maximum = profile[input_spec.name]["max_shape"]
                if len(minimum) != len(input_spec.min_shape):
                    raise ValueError(f"TensorRT profile rank does not match artifact input {input_spec.name!r}")
                for axis, (profile_min, profile_max, graph_min, graph_max) in enumerate(
                    zip(minimum, maximum, input_spec.min_shape, input_spec.max_shape, strict=True)
                ):
                    if profile_min < graph_min or profile_max > graph_max:
                        raise ValueError(
                            f"TensorRT profile {profile_index} input {input_spec.name!r} axis {axis} range "
                            f"[{profile_min}, {profile_max}] exceeds the recorded graph range "
                            f"[{graph_min}, {graph_max}]; output bounds cannot be established"
                        )

    def _artifact_profiles(self) -> tuple[dict[str, dict[str, tuple[int, ...]]], ...]:
        """Describe exact profile ranges as plain data in engine input order."""
        if not self._trt_optimization_profiles or not self._input_names:
            raise RuntimeError("TensorRT artifact requires its optimization profiles")

        result = []
        for profile in self._trt_optimization_profiles:
            unknown_names = set(profile) - set(self._input_names)
            missing_names = set(self._input_names) - set(profile)
            if unknown_names or missing_names:
                raise ValueError(
                    f"TensorRT profile inputs do not match the engine (missing: {sorted(missing_names)}, "
                    f"unknown: {sorted(unknown_names)})"
                )
            result.append({name: self._artifact_profile_shapes(name, profile[name]) for name in self._input_names})
        return tuple(result)

    @staticmethod
    def _artifact_profile_shapes(name: str, shape_range: Any) -> dict[str, tuple[int, ...]]:
        """Validate and copy a Polygraphy shape range into portable metadata."""
        minimum, optimum, maximum = tuple(shape_range.min), tuple(shape_range.opt), tuple(shape_range.max)
        if len({len(minimum), len(optimum), len(maximum)}) != 1:
            raise ValueError(f"TensorRT profile shapes for {name!r} must have the same rank")
        if any(
            not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0
            for shape in (minimum, optimum, maximum)
            for dimension in shape
        ):
            raise ValueError(f"TensorRT profile shapes for {name!r} must contain positive integers")
        if any(not low <= opt <= high for low, opt, high in zip(minimum, optimum, maximum, strict=True)):
            raise ValueError(f"TensorRT profile shapes for {name!r} must satisfy min <= opt <= max")
        return {"min_shape": minimum, "opt_shape": optimum, "max_shape": maximum}

    def _deactivate(self):
        """Deactivate the TensorRT engine."""
        logger.debug("Deactivating TensorRT backend")
        self._trt_runtime = None

        try:
            if self._cuda_stream is not None:
                self._cuda_stream.synchronize()
            self._cuda_graphs.clear()
            with contextlib.ExitStack() as stack:
                if self._context:
                    stack.enter_context(self._context)

            if self._output_allocator is not None:
                self._output_allocator.clear()
            if self._base_output_allocator is not None:
                self._base_output_allocator.clear()
            self._base_context = None
            self._base_output_allocator = None

            # Retain tensor names and profiles for artifact generation after deactivation.
            for attr_name in [
                "_io_tensors",
                "_engine_info",
                "_cuda_stream",
                "_start_time",
                "_end_time",
                "_outputs",
                "_context",
                "_output_allocator",
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

    def _save_trt_optimization_profiles(self, optimization_profiles: list[Profile], cache_dir: Path) -> Path:
        """Save the TensorRT optimization profiles to a file."""
        optimization_profiles_path = self._prepare_trt_optimization_profiles_path(cache_dir)
        with open(optimization_profiles_path, "w") as f:
            json.dump([TensorRTProfile.profile_to_dict(profile) for profile in optimization_profiles], f)
        logger.info("TensorRT optimization profiles saved to %s", optimization_profiles_path)
        return optimization_profiles_path

    def _load_trt_optimization_profiles(self, trt_optimization_profiles_path: Path) -> list[Profile]:
        """Load the TensorRT optimization profiles from a file."""
        try:
            with Path(trt_optimization_profiles_path).open("r") as f:
                profiles_json = json.load(f)
                profiles = [Profile() for _ in profiles_json]
                for profile, profile_json in zip(profiles, profiles_json, strict=True):
                    for name, (min_, opt_, max_) in profile_json.items():
                        profile.add(name, tuple(min_), tuple(opt_), tuple(max_))
                return profiles
        except Exception as e:
            logger.debug("Failed to load TensorRT optimization profiles: %s", e)
            # profile 0 is used by default
            return []

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
            name = GraphSpec.tensor_name(locator, tensor_spec, "output")
            if name in outputs:
                result = locator.set_value(result, outputs[name].clone())
                del outputs[name]
            else:
                logger.debug("Output: %s not found in outputs", name)

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
        forward_inputs = self._graph_spec.forward_signature.normalize(args, kwargs)
        for locator, tensor_spec in self._graph_spec.input_spec.tensor_data:
            name = GraphSpec.tensor_name(locator, tensor_spec, "input")
            if name in engine_input_names:
                inputs[name] = locator.get_value(forward_inputs.arguments)
            else:
                logger.debug("Input: %s not found in inputs", name)

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
            if self._use_cuda_graphs:
                tensor = self._cuda_graphs.copy_input(name, tensor, self._cuda_stream)
                if self._cuda_graphs.active.graph is not None:
                    # Replaying must not modify the context captured by this graph.
                    continue

            success = self._context.set_input_shape(name, shape)
            if not success:
                engine_shape = tuple(self._io_tensors[name]["shape"])
                profile_index = self._context.active_optimization_profile
                raise RuntimeError(
                    f"TensorRT rejected shape {shape} for input {name!r} with optimization profile {profile_index}; "
                    f"engine shape is {engine_shape}"
                )
            logger.debug("Set input shape for %s successfully", name)

            success = self._context.set_tensor_address(name, tensor.data_ptr())
            if not success:
                raise RuntimeError(f"TensorRT rejected the address for input {name!r}")
            logger.debug("Set tensor address for %s successfully", name)

    def _set_optimization_profiles(self, inputs: dict[str, torch.Tensor]) -> None:
        """Select an ordinary context or the static profile's cached graph context."""
        index = self._find_optimization_profile(inputs)
        self._cuda_graphs.active = None
        # Drop references to the previous graph context before the cache can evict it.
        if self._base_context is not None:
            self._context = self._base_context
            self._output_allocator = self._base_output_allocator
        if self._config.use_cuda_graphs and self._cuda_graphs.is_eligible(index):
            profile = self._cuda_graphs.select(
                index, self._base_context.engine, self._cuda_stream, self._create_output_allocator
            )
            if profile is not None:
                self._context = profile.context
                self._output_allocator = profile.output_allocator
                return

        if self._trt_optimization_profiles and self._context.active_optimization_profile != index:
            if not self._context.set_optimization_profile_async(index, self._cuda_stream.cuda_stream):
                raise RuntimeError(f"TensorRT rejected optimization profile {index}")

    def _find_optimization_profile(self, inputs: dict[str, torch.Tensor]) -> int:
        """Find the first matching profile, including validation for a single profile.

        Args:
            inputs: Dictionary mapping input names to tensors
        """
        if not self._trt_optimization_profiles:
            return 0

        for idx, profile in enumerate(self._trt_optimization_profiles):
            if profile.keys() != inputs.keys():
                continue

            for name, (min_shape, _, max_shape) in profile.items():
                tensor = inputs[name]
                if len(tensor.shape) != len(min_shape):
                    break
                if not all(
                    min_ <= actual <= max_
                    for (actual, min_, max_) in zip(tensor.shape, min_shape, max_shape, strict=True)
                ):
                    break

            else:
                logger.debug("Selected TensorRT optimization profile %d", idx)
                return idx

        raise RuntimeError("No TensorRT optimization profile matches the input shapes")

    def get_profiles(self, graph_spec: GraphSpec, samples: Sequence[Sample]) -> list[Profile]:
        """Create profiles from samples or from graph_spec.

        If self._config.profiles is a list, return the user provided profiles.
        If self._config.profiles is ProfileMode.SINGLE, create a single profile from the graph spec.
        If self._config.profiles is ProfileMode.SAMPLES_USED, create profiles from shapes seen in samples.
        Explicit module shape definitions always produce a single authoritative profile and cannot be combined with
        user-provided TensorRT profiles.

        Args:
            graph_spec: Input graph spec
            samples: Recorded samples for the model.

        Returns:
            List of The Polygraphy Profile objects
        """
        if graph_spec.dynamic_shapes:
            if isinstance(self._config.profiles, list):
                raise AITuneUserInputError(
                    "TensorRT profiles cannot be provided together with module dynamic shape definitions."
                )
            return self._get_profiles_from_shapes()

        # if user provided profiles, return them
        if isinstance(self._config.profiles, list):
            return self._get_user_profiles(graph_spec)

        if self._config.profiles == ProfileMode.SINGLE:
            # this will create a single profile from the graph spec
            return self._get_profiles_from_shapes()

        profiles = OrderedDict()
        logger.info("Creating profiles from samples used for tuning")

        # Create a profile
        for idx, sample in enumerate(samples):
            profile = TensorRTProfile()
            args, kwargs = sample
            forward_inputs = graph_spec.forward_signature.normalize(args, kwargs)
            for locator, tensor_spec in graph_spec.input_spec.tensor_data:
                shape = locator.get_value(forward_inputs.arguments).shape
                profile.profile.add(GraphSpec.tensor_name(locator, tensor_spec, "input"), shape, shape, shape)

            logger.debug("Created profile %d: %s", idx, profile)
            profiles[profile] = True

        return [profile.profile for profile in profiles.keys()]

    def _get_user_profiles(self, graph_spec: GraphSpec) -> list[Profile]:
        """Resolve public profile paths to graph tensor names and validate the bindings."""
        bindings = {
            format_tensor_name(locator.path, "input"): GraphSpec.tensor_name(locator, spec, "input")
            for locator, spec in graph_spec.input_spec.tensor_data
        }
        input_names = set(bindings)
        for profile in self._config.profiles:
            profile_names = set(profile.profile)
            missing_names = input_names - profile_names
            unknown_names = profile_names - input_names
            if missing_names or unknown_names:
                raise AITuneUserInputError(
                    f"TensorRT profile inputs {sorted(profile_names)} do not match recorded tensor inputs "
                    f"{sorted(input_names)} (missing: {sorted(missing_names)}, unknown: {sorted(unknown_names)})."
                )
        profiles = []
        for user_config in self._config.profiles:
            resolved = Profile()
            for name, (minimum, optimal, maximum) in user_config.profile.items():
                resolved.add(bindings[name], minimum, optimal, maximum)
            profiles.append(resolved)
        return profiles

    def _get_profiles_from_shapes(self) -> list[Profile]:
        """Create TensorRT optimization profiles.

        Returns:
            List of Polygraphy Profile objects
        """
        profiles = []

        min_shapes, opt_shapes, max_shapes = self._get_shapes(self._graph_spec)

        if any([min_shapes, opt_shapes, max_shapes]):
            profile = Profile()
            for name, min_shape in (min_shapes or {}).items():
                opt_shape = opt_shapes.get(name) if opt_shapes else min_shape
                max_shape = max_shapes.get(name) if max_shapes else opt_shape
                logger.info("Adding profile for '%s': min=%s, opt=%s, max=%s", name, min_shape, opt_shape, max_shape)
                profile.add(name=name, min=min_shape, opt=opt_shape, max=max_shape)
            profiles = [profile]

            logger.info("Single profile created from graph spec shapes")

        return profiles

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
        for locator, tensor_spec in graph_spec.input_spec.tensor_data:
            input_name = GraphSpec.tensor_name(locator, tensor_spec, "input")
            min_shape, opt_shape, max_shape = graph_spec.get_effective_input_shapes(locator, tensor_spec)
            min_shapes[input_name] = tuple(min_shape)
            opt_shapes[input_name] = tuple(opt_shape)
            max_shapes[input_name] = tuple(max_shape)

        return min_shapes, opt_shapes, max_shapes

    @property
    def _use_cuda_graphs(self) -> bool:
        """Use graphs only for fixed-shape profiles, until capture fails for this instance."""
        return (
            self._config.use_cuda_graphs
            and not self._cuda_graphs.capture_failed
            and self._cuda_graphs.active is not None
        )

    def _execute_engine(self):
        """Execute TensorRT normally, propagating engine execution failures."""
        if not self._context.execute_async_v3(self._cuda_stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")

    def to_dict(self):
        """Returns the state_dict of the backend."""
        return {
            self.STATE_TYPE: self.__class__.__name__,
            self.STATE_ENGINE_PATH: self._engine_artifact,
            self.STATE_OUTPUT_OBJECT: self._output_object,
            self.STATE_GRAPH_SPEC: self._graph_spec.to_dict(),
            self.STATE_DEVICE: self._device,
            self.STATE_QUANTIZATION_CONFIG: self._config.quantization_config,
            self.STATE_CONFIG: self._config.to_dict(),
            self.STATE_USE_CUDA_GRAPHS: self._config.use_cuda_graphs,
            self.STATE_TRT_OPTIMIZATION_PROFILES_PATH: self._trt_optimization_profiles_artifact,
            self.STATE_SAMPLES: (self._samples.to_dict() if isinstance(self._samples, SampleStore) else self._samples),
        }

    @classmethod
    def from_dict(cls, module: torch.nn.Module, state_dict: dict):
        """Creates a backend from a state_dict."""
        backend = cls()
        backend._engine_artifact = state_dict[cls.STATE_ENGINE_PATH]
        backend._trt_optimization_profiles_artifact = state_dict[cls.STATE_TRT_OPTIMIZATION_PROFILES_PATH]
        backend._graph_spec = GraphSpec.from_dict(state_dict[cls.STATE_GRAPH_SPEC])
        backend._device = state_dict[cls.STATE_DEVICE]
        samples_state = state_dict.get(cls.STATE_SAMPLES)
        backend._samples = SampleStore.from_dict(samples_state) if isinstance(samples_state, dict) else samples_state
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
