# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TensorRT backend."""

import contextlib
import copy
import json
import logging
import shutil
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, cast

import torch
import torch.nn as nn
from polygraphy.backend.trt import Profile
from polygraphy.logger import G_LOGGER

from aitune.exceptions import AITuneUserInputError
from aitune.torch.backend.backend import (
    Backend,
    BackendBuildStep,
    BackendConfig,
    BackendState,
    BuildMode,
    ExecutionMode,
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
    _execution_modes = frozenset({ExecutionMode.SINGLE_GPU})

    # State dictionary keys
    STATE_TYPE = "type"
    STATE_ENGINE_PATH = "engine_path"
    STATE_TRT_OPTIMIZATION_PROFILES_PATH = "trt_optimization_profiles_path"
    STATE_OUTPUT_OBJECT = "output_object"
    STATE_ONNX_INPUT_NAMES = "onnx_input_names"
    STATE_GRAPH_SPEC = "graph_spec"
    STATE_DEVICE = "device"
    STATE_QUANTIZATION_CONFIG = "quantization_config"
    STATE_CONFIG = "config"
    STATE_USE_CUDA_GRAPHS = "use_cuda_graphs"

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
        self._onnx_input_names: dict[str, str] | None = None

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

        cuda_set_device(self._device)
        if isinstance(module, OnnxModule):
            import onnx

            module.deactivate()
            if isinstance(self._config.quantization_config, TorchQuantizationConfig):
                raise AITuneUserInputError(
                    "Torch quantization requires a Torch module; use ONNX quantization for OnnxModule."
                )
            graph = onnx.load(module.path, load_external_data=False).graph
            initializers = {tensor.name for tensor in graph.initializer}
            names = [node.name for node in graph.input if node.name not in initializers]
            del graph
            args, kwargs = samples[0]
            # Normalize names in place of tensors to map recorded paths to ONNX inputs.
            named_inputs = graph_spec.forward_signature.normalize(
                tuple(names[: len(args)]), {name: name for name in kwargs}
            )
            self._onnx_input_names = {
                format_tensor_name(locator.path, "input"): locator.get_value(named_inputs.arguments)
                for locator, _ in graph_spec.input_spec.tensor_data
            }
        else:
            self._output_object = self._get_output_object(module=module, sample=samples[0])

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
                self._trt_optimization_profiles = self._get_engine_profiles(graph_spec=graph_spec, samples=samples)

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
                    input_names=self._onnx_input_names,
                )
                result["onnx_size_bytes"] = onnx_path_quantized.stat().st_size

            step = TensorRTBuildStep.TENSORRT_ENGINE_BUILD
            with annotate(step.annotation), self._track_build_step(step) as result:
                # Initialize TensorRT builder
                logger.info("Initializing TensorRT builder")
                engine_path = self._prepare_trt_engine_path(cache_dir)
                self._trt_optimization_profiles = self._get_engine_profiles(graph_spec=graph_spec, samples=samples)

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
                    input_names=self._onnx_input_names,
                )
                result["onnx_size_bytes"] = onnx_path_autocasted.stat().st_size

            step = TensorRTBuildStep.TENSORRT_ENGINE_BUILD
            with annotate(step.annotation), self._track_build_step(step) as result:
                # Initialize TensorRT builder
                logger.info("Initializing TensorRT builder")
                engine_path = self._prepare_trt_engine_path(cache_dir)
                self._trt_optimization_profiles = self._get_engine_profiles(graph_spec=graph_spec, samples=samples)

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
                path, size = self._copy_or_not_onnx_module(module, path)
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

    def _copy_or_not_onnx_module(self, module: OnnxModule, path: Path) -> tuple[Path, int]:
        """Copy the ONNX model to the cache directory."""
        import onnx
        from onnx.external_data_helper import _get_all_tensors

        # model might be big and copying it might not be desirable
        if global_config.disable_onnx_model_copy:
            return (Path(module.path), module.path.stat().st_size)

        model = onnx.load(module.path, load_external_data=False)
        locations = {
            entry.value
            for tensor in _get_all_tensors(model)
            for entry in tensor.external_data
            if entry.key == "location"
        }
        size = module.path.stat().st_size
        for location in locations:
            destination = path.parent / location
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(module.path.parent / location, destination)
            size += destination.stat().st_size

        shutil.copy2(module.path, path)
        return (path, size)

    def _get_engine_profiles(self, graph_spec: GraphSpec, samples: Sequence[Sample]) -> list[Profile]:
        """Map recorded profile names to native ONNX inputs when needed."""
        profiles = self.get_profiles(graph_spec, samples)
        if self._onnx_input_names is None:
            return profiles
        native_profiles = []
        for profile in profiles:
            native_profile = Profile()
            for name, (min_shape, opt_shape, max_shape) in profile.items():
                native_profile.add(self._onnx_input_names[name], min_shape, opt_shape, max_shape)
            native_profiles.append(native_profile)
        return native_profiles

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
                self._trt_optimization_profiles = self._get_engine_profiles(graph_spec=graph_spec, samples=samples)

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
        if self._onnx_input_names is not None:
            return {name: tensor.clone() for name, tensor in outputs.items()}

        result = copy.deepcopy(self._output_object)  # make each results a unique copy of the original output object
        for locator, _ in self._graph_spec.output_spec.tensor_data:
            name = format_tensor_name(locator.path, "output")
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
        for locator, _ in self._graph_spec.input_spec.tensor_data:
            name = format_tensor_name(locator.path, "input")
            if self._onnx_input_names is not None:
                name = self._onnx_input_names[name]
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
            input_names = {
                format_tensor_name(locator.path, "input") for locator, _ in graph_spec.input_spec.tensor_data
            }
            for profile in self._config.profiles:
                profile_names = set(profile.profile)
                missing_names = input_names - profile_names
                unknown_names = profile_names - input_names
                if missing_names or unknown_names:
                    raise AITuneUserInputError(
                        f"TensorRT profile inputs {sorted(profile_names)} do not match recorded tensor inputs "
                        f"{sorted(input_names)} (missing: {sorted(missing_names)}, unknown: {sorted(unknown_names)})."
                    )
            return [user_config.profile for user_config in self._config.profiles]

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
            for locator, _ in graph_spec.input_spec.tensor_data:
                shape = locator.get_value(forward_inputs.arguments).shape
                profile.add_input_shape(locator.path, shape, shape, shape)

            logger.debug("Created profile %d: %s", idx, profile)
            profiles[profile] = True

        return [profile.profile for profile in profiles.keys()]

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
            input_name = format_tensor_name(locator.path, "input")
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
            self.STATE_ONNX_INPUT_NAMES: self._onnx_input_names,
            self.STATE_GRAPH_SPEC: self._graph_spec.to_dict(),
            self.STATE_DEVICE: self._device,
            self.STATE_QUANTIZATION_CONFIG: self._config.quantization_config,
            self.STATE_CONFIG: self._config.to_dict(),
            self.STATE_USE_CUDA_GRAPHS: self._config.use_cuda_graphs,
            self.STATE_TRT_OPTIMIZATION_PROFILES_PATH: self._trt_optimization_profiles_artifact,
        }

    @classmethod
    def from_dict(cls, module: torch.nn.Module, state_dict: dict):
        """Creates a backend from a state_dict."""
        backend = cls()
        backend._engine_artifact = state_dict[cls.STATE_ENGINE_PATH]
        backend._trt_optimization_profiles_artifact = state_dict[cls.STATE_TRT_OPTIMIZATION_PROFILES_PATH]
        backend._graph_spec = GraphSpec.from_dict(state_dict[cls.STATE_GRAPH_SPEC])
        backend._device = state_dict[cls.STATE_DEVICE]
        backend.state = BackendState.CHECKPOINT_LOADED

        # Reconstruct config with quantization settings
        backend._config = TensorRTBackendConfig.from_dict(state_dict[cls.STATE_CONFIG])

        backend._output_object = state_dict[cls.STATE_OUTPUT_OBJECT]
        backend._onnx_input_names = state_dict.get(cls.STATE_ONNX_INPUT_NAMES)
        # Ensure CUDA graphs are disabled when loading from checkpoint
        # CUDA graphs cannot be serialized and must be re-captured
        if backend._config.use_cuda_graphs:
            logger.info("CUDA graphs were enabled in saved state, but will be re-captured on first inference")
            # CUDA graph state will be None initially, triggering re-capture

        return backend
