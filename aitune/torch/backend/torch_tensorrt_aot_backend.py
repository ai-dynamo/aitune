# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TorchTensorRT backend with AOT compilation and intermediate model save."""

import operator
from dataclasses import asdict, dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar, cast

import torch
import torch.nn as nn
from torch._export.verifier import SpecViolationError

from aitune.torch.backend.backend import (
    Backend,
    BackendBuildStep,
    BackendConfig,
    BackendState,
    BuildMode,
    ExecutionMode,
)
from aitune.torch.backend.torch_tensorrt_logging import torch_tensorrt_errors, torch_tensorrt_warnings
from aitune.torch.checkpoint.artifact import ArtifactPath
from aitune.torch.distributed import distributed_output_path
from aitune.torch.libs.exporting import TorchExporter
from aitune.torch.libs.exporting.export_validation import preflight_export, validate_export_profiles
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import SampleStore
from aitune.torch.utils.cuda_utils import assert_is_available as assert_cuda_is_available
from aitune.torch.utils.cuda_utils import get_device as get_cuda_device
from aitune.torch.utils.module import move_module_to_device
from aitune.torch.utils.tensor import format_tensor_name

try:
    import torch_tensorrt
    from torch_tensorrt.dynamo import CompilationSettings as TorchTensorRTConfig
except (ImportError, RuntimeError, OSError):
    torch_tensorrt = None

    @dataclass
    class TorchTensorRTConfig:
        """Fallback settings matching upstream defaults when torch_tensorrt is unavailable.

        Attributes:
            enabled_precisions: Enabled TensorRT precisions. Defaults to None.
            use_explicit_typing: Whether to use explicit typing. Defaults to True.
            workspace_size: TensorRT workspace size. Defaults to 0.
        """

        enabled_precisions: set[torch.dtype] | None = None
        use_explicit_typing: bool = True
        workspace_size: int = 0


logger = getLogger(__name__)


class TorchTensorRTSerializationContractError(RuntimeError):
    """Raised when serialization would change a compiled program's exported contract."""


class TorchTensorRTAotBuildStep(BackendBuildStep):
    """Identifiers for discrete sub-steps of a TorchTensorRTAot backend build."""

    TORCH_EXPORT = "Torch export"
    TORCHTRT_COMPILE = "Torch-TensorRT compile"
    COMPILED_MODEL_SAVE = "Compiled model save"


def assert_torch_tensorrt():
    """Check if torch_tensorrt is installed."""
    if torch_tensorrt is None:
        raise ImportError("torch-tensorrt is not installed. Please install `torch-tensorrt`.")


@dataclass
class TorchTensorRTAotBackendConfig(BackendConfig):
    """Configuration for TorchTensorRTAotBackend.

    See torch_tensorrt/dynamo/_settings.py CompilationSettings for compile_config(TorchTensorRTConfig)

    Attributes:
        compile_config: Torch-TensorRT compilation settings. Defaults to TorchTensorRTConfig().
        pickle_protocol: Pickle protocol for saved compiled artifacts. Defaults to 5.
    """

    compile_config: TorchTensorRTConfig = field(  # pytype: disable=invalid-annotation
        default_factory=TorchTensorRTConfig
    )
    pickle_protocol: int = 5

    def describe(self) -> str:
        """Describe the backend configuration. Display only changed fields."""
        other = self.__class__()
        compile_config_parts = self._get_changed_fields(
            self.compile_config,
            other.compile_config,
            exclude=["device"],
            include=["enabled_precisions"],
        )
        parts = [f"compile_config=TorchTensorRTConfig({','.join(compile_config_parts)})"]

        changed_fields = self._get_changed_fields(self, other, exclude=["compile_config"])
        parts.extend(changed_fields)

        return ",".join(parts)

    def to_dict(self):
        """Saves the backend configuration to a file."""
        state_dict = asdict(self)
        state_dict["compile_config"] = asdict(self.compile_config)  # explicitly convert to dict
        return state_dict

    @classmethod
    def from_dict(cls, data: dict) -> "TorchTensorRTAotBackendConfig":
        """Initialise config from a plain dict (e.g. parsed from YAML).

        ``compile_config`` may be passed as a nested dict and will be
        reconstructed into a ``TorchTensorRTConfig`` instance automatically.
        """
        data = dict(data)
        data.pop("ir", None)
        if isinstance(data.get("compile_config"), dict):
            data["compile_config"] = TorchTensorRTConfig(**data["compile_config"])
        return cls(**data)


class TorchTensorRTAotBackend(Backend):
    """Backend that compiles model using TensorRT."""

    _build_mode = BuildMode.AHEAD_OF_TIME
    _execution_modes = frozenset({ExecutionMode.SINGLE_GPU, ExecutionMode.MULTI_GPU})

    # State dictionary keys
    STATE_TYPE = "type"
    EXPORTED_MODEL_PATH_KEY = "exported_model_path"
    STATE_DEVICE = "device"
    STATE_GRAPH_SPEC = "graph_spec"

    # Supported devices
    _devices: ClassVar[list[str]] = ["cuda"]

    def __init__(
        self,
        config: TorchTensorRTAotBackendConfig | None = None,
    ):
        """Initialize TensorRT backend.

        Args:
            config: Configuration for TensorRT compilation
        """
        super().__init__()
        assert_cuda_is_available()
        assert_torch_tensorrt()

        # initialize variables
        self._config = config or TorchTensorRTAotBackendConfig()

        # build variables
        self._opt_module = None
        self._exported_model_artifact: ArtifactPath | None = None
        self._graph_spec: GraphSpec | None = None

    def key(self) -> str:
        """Returns the key of the backend."""
        return f"{self.__class__.__name__}_{self._config.key()}"

    def describe(self) -> str:
        """Returns the description of the backend."""
        return f"{self.__class__.__name__}({self._config.describe()})"

    def _compile_settings(self) -> dict[str, Any]:
        """Copy compile settings and isolate the timing cache by rank."""
        settings = asdict(self._config.compile_config)
        timing_cache_path = settings.get("timing_cache_path")
        if not timing_cache_path:
            return settings

        timing_cache = Path(timing_cache_path)
        resolved_path = distributed_output_path(timing_cache)
        if resolved_path != timing_cache:
            logger.info("Using rank-isolated Torch-TensorRT timing cache: %s", resolved_path)
        settings["timing_cache_path"] = resolved_path.as_posix()
        return settings

    def _build(
        self,
        module: nn.Module,
        graph_spec: GraphSpec,
        samples: SampleStore,
        cache_dir: Path,
    ) -> Backend:
        """Build the model with TensorRT.

        Exports the module via ``torch.export.export`` with shared ``Dim`` instances
        across inputs, then hands the resulting ``ExportedProgram`` to
        ``torch_tensorrt.dynamo.compile``. This replaces the previous flow that called
        ``torch_tensorrt.compile`` and let it run its own export, which built one
        independent ``Dim`` per ``Input`` and raised ``ConstraintViolationError``
        whenever two inputs had to share a runtime axis (e.g. HF encoders' shared
        batch on ``input_ids`` and ``attention_mask``).

        Workaround pending a ``dynamic_shapes`` passthrough on ``torch_tensorrt.compile``;
        revert to ``torch_tensorrt.compile(...)`` once that lands upstream.
        """
        logger.info("Start compiling torch module with Torch-TensorRT using torch_tensorrt.dynamo.compile.")

        model = module.eval()
        move_module_to_device(model, self._device)
        self._graph_spec = graph_spec

        # Set the device for the TensorRT backend compilation
        self._config.compile_config.device = torch_tensorrt.Device(get_cuda_device(self._device))

        with self._track_build_step(TorchTensorRTAotBuildStep.TORCH_EXPORT):
            # ``use_auto=False`` creates finite ranges required by TensorRT optimization profiles.
            export_result = TorchExporter(
                use_auto=False,
                strict=False,
                fallback_to_dynamic_hints=True,
            ).export(
                model,
                samples[0],
                graph_spec,
                device=self._device,
            )
            exported = export_result.exported_program
            preflight_export(exported)
            validate_export_profiles(exported, graph_spec)

        # Optimization-profile inputs for TRT engine building.
        input_signature = []
        for locator, tensor_spec in graph_spec.input_spec.tensor_data:
            input_name = format_tensor_name(locator.path, "input")
            min_shape, opt_shape, max_shape = graph_spec.get_effective_input_shapes(locator, tensor_spec)
            input_i = torch_tensorrt.Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=tensor_spec.dtype,
                name=input_name,
            )
            input_signature.append(input_i)
            logger.info(
                "Torch-TensorRT input profile %s: min=%s opt=%s max=%s dtype=%s",
                input_name,
                min_shape,
                opt_shape,
                max_shape,
                tensor_spec.dtype,
            )

        with self._track_build_step(TorchTensorRTAotBuildStep.TORCHTRT_COMPILE):
            logger.info("Compile model with Torch-TensorRT.")
            with torch_tensorrt_warnings(torch_tensorrt):
                trt_model_compiled = torch_tensorrt.dynamo.compile(
                    exported,
                    inputs=input_signature,
                    **self._compile_settings(),
                )

        if self.stateful_validation is not None:
            self._opt_module = trt_model_compiled
            try:
                self.stateful_validation.check(self._infer, self._device, name="Torch-TensorRT before serialization")
            finally:
                self._opt_module = None

        with self._track_build_step(TorchTensorRTAotBuildStep.COMPILED_MODEL_SAVE) as result:
            self._exported_model_artifact = self._create_exported_model_artifact(cache_dir)
            _populate_mutation_output_metadata(trt_model_compiled, exported, graph_spec)
            _save_compiled_model(
                trt_model_compiled,
                self._exported_model_artifact.path,
                input_signature,
                export_result.dynamic_shapes,
                self._config.pickle_protocol,
                allow_retrace=not (
                    graph_spec.forward_contract.input_mutations
                    or graph_spec.forward_contract.output_input_aliases
                    or exported.graph_signature.buffers_to_mutate
                    or exported.graph_signature.user_inputs_to_mutate
                ),
            )
            result["compiled_model_size_bytes"] = self._exported_model_artifact.path.stat().st_size

        # Validate the same persisted artifact that production will load.
        del trt_model_compiled
        self._activate()
        logger.info("Module has been compiled, saved, and reloaded with TensorRT.")
        return self

    def _activate(self):
        """Load compiled module."""
        exported_model_artifact = cast(ArtifactPath, self._exported_model_artifact)
        with torch_tensorrt_errors(torch_tensorrt):
            loaded = torch_tensorrt.load(exported_model_artifact.path.as_posix())
            self._opt_module = loaded.module().to(self._device)

    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        """Run the flat Torch-TensorRT module, then restore outputs and input changes."""
        if self._opt_module is None:
            raise RuntimeError("Module is not activated. Please call activate() first.")

        graph_spec = cast(GraphSpec, self._graph_spec)
        with torch.no_grad():
            forward_inputs = graph_spec.forward_signature.normalize(args, kwargs)
            flat_inputs = graph_spec.input_spec.flatten_tensors(forward_inputs.arguments)
            flat_outputs = self._opt_module(*flat_inputs)
            return graph_spec.restore_outputs(flat_outputs, forward_inputs.arguments)

    def _deactivate(self):
        """Deactivate backend and cleanup."""
        self._opt_module = None

    def _deploy(self):
        """Deploys the backend."""
        self._activate()

    def _create_exported_model_artifact(self, cache_dir: Path) -> ArtifactPath:
        """Create the exported model artifact.

        In order to avoid name clashes we create a unique path for each model and graph spec.
        """
        cache_dir.mkdir(parents=True, exist_ok=True)
        return ArtifactPath(cache_dir, "exported_model.pt2")

    def to_dict(self) -> dict:
        """Returns the state_dict of the backend."""
        if self._exported_model_artifact is None or self._graph_spec is None:
            raise RuntimeError("No exported model path available. Model must be built first.")
        return {
            self.STATE_TYPE: self.__class__.__name__,
            self.EXPORTED_MODEL_PATH_KEY: self._exported_model_artifact,
            self.STATE_DEVICE: self._device,
            self.STATE_GRAPH_SPEC: self._graph_spec.to_dict(),
        }

    @classmethod
    def from_dict(cls, module: torch.nn.Module | None, state_dict: dict) -> "TorchTensorRTAotBackend":
        """Creates a backend from a state_dict."""
        if state_dict.get(cls.STATE_TYPE) != cls.__name__:
            raise ValueError(f"Invalid state_dict type: {state_dict.get(cls.STATE_TYPE)}")

        backend = cls()  # create with default args, config is not needed
        backend._exported_model_artifact = state_dict[cls.EXPORTED_MODEL_PATH_KEY]
        backend._device = state_dict[cls.STATE_DEVICE]
        backend._graph_spec = GraphSpec.from_dict(state_dict[cls.STATE_GRAPH_SPEC])
        backend.state = BackendState.CHECKPOINT_LOADED
        return backend


def _save_compiled_model(
    module: nn.Module,
    path: Path,
    inputs: list[Any],
    dynamic_shapes: dict[str, Any] | None,
    pickle_protocol: int,
    *,
    allow_retrace: bool = True,
) -> None:
    """Save a compiled graph and retry known direct-export failures with retracing."""
    _validate_exported_program(module)
    try:
        with torch_tensorrt_errors(torch_tensorrt):
            torch_tensorrt.save(
                module,
                path.as_posix(),
                retrace=False,
                pickle_protocol=pickle_protocol,
            )
        logger.info("Saved the Torch-TensorRT artifact without retracing.")
    except (SpecViolationError, RuntimeError) as error:
        if not allow_retrace:
            raise TorchTensorRTSerializationContractError(
                "Direct Torch-TensorRT serialization failed for a model that changes inputs or shares outputs "
                "with inputs. Retracing is disabled because it may lose these input changes or shared objects."
            ) from error
        integer_parameter_error = "Only Tensors of floating point and complex dtype can require gradients"
        if isinstance(error, RuntimeError) and integer_parameter_error not in str(error):
            raise
        logger.warning(
            "Direct Torch-TensorRT serialization failed; retrying with retrace enabled.",
            exc_info=True,
        )
        with torch_tensorrt_errors(torch_tensorrt):
            torch_tensorrt.save(
                module,
                path.as_posix(),
                retrace=True,
                inputs=inputs,
                dynamic_shapes=dynamic_shapes,
                pickle_protocol=pickle_protocol,
            )
        logger.info("Saved the Torch-TensorRT artifact using the retracing fallback.")


def _validate_exported_program(module: Any) -> None:
    """Validate an ExportedProgram when the supplied Torch-TensorRT object exposes that API."""
    validate = getattr(module, "validate", None)
    if not callable(validate):
        return
    try:
        validate()
    except Exception as error:
        raise TorchTensorRTSerializationContractError(
            "Torch-TensorRT produced an invalid exported program before serialization."
        ) from error


def _graph_module(module: Any) -> Any | None:
    """Return the FX graph module contained in an exported or compiled object."""
    graph_module = getattr(module, "graph_module", None)
    if graph_module is not None:
        return graph_module
    return module if hasattr(module, "graph") else None


def _flat_output_nodes(module: Any) -> tuple[Any, ...]:
    """Return tensor-producing nodes referenced by the graph output."""
    graph_module = _graph_module(module)
    if graph_module is None:
        return ()
    output_nodes = [node for node in graph_module.graph.nodes if node.op == "output"]
    if len(output_nodes) != 1:
        return ()

    flattened = []

    def visit(value):
        if hasattr(value, "meta"):
            flattened.append(value)
        elif isinstance(value, tuple | list):
            for item in value:
                visit(item)

    visit(output_nodes[0].args)
    return tuple(flattened)


def _populate_mutation_output_metadata(compiled: Any, exported: Any, graph_spec: GraphSpec) -> None:
    """Copy valid Torch Export metadata onto Torch-TensorRT's generated mutation outputs."""
    _populate_copy_node_metadata(compiled)
    mutations = graph_spec.extra_tensor_mutations()
    if not mutations:
        return
    source_outputs = _flat_output_nodes(exported)
    compiled_outputs = _flat_output_nodes(compiled)
    for mutation in mutations:
        if mutation.output_index is None:
            raise TorchTensorRTSerializationContractError(f"Mutation output {mutation.locator} has no output index.")
        index = mutation.output_index
        if index >= len(source_outputs) or index >= len(compiled_outputs):
            raise TorchTensorRTSerializationContractError(
                f"Mutation output {mutation.locator} at index {index} is missing from the compiled graph."
            )
        source = source_outputs[index]
        target = compiled_outputs[index]
        if "val" not in source.meta:
            raise TorchTensorRTSerializationContractError(
                f"Mutation output {mutation.locator} at index {index} has no Torch Export value metadata."
            )
        target.meta["val"] = source.meta["val"]
        if target.op == "call_function" and target.target == operator.getitem:
            producer, output_index = target.args[:2]
            producer_value = producer.meta.get("val")
            if isinstance(output_index, int) and isinstance(producer_value, tuple | list):
                values = list(producer_value)
                while len(values) <= output_index:
                    values.append(source.meta["val"])
                values[output_index] = source.meta["val"]
                producer.meta["val"] = type(producer_value)(values)


def _populate_copy_node_metadata(compiled: Any) -> None:
    """Derive missing metadata for in-place copy nodes from their mutation targets."""
    graph_module = _graph_module(compiled)
    if graph_module is None:
        return
    for node in graph_module.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.copy_.default or "val" in node.meta:
            continue
        target = node.args[0]
        target_value = getattr(target, "meta", {}).get("val")
        if target_value is None:
            raise TorchTensorRTSerializationContractError(
                f"Mutation node {node.name} has no value metadata and its copy target cannot supply it."
            )
        node.meta["val"] = target_value
