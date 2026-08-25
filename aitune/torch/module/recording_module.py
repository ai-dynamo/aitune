# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inplace wrap modules."""

import itertools
import logging
import tempfile
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import torch

from aitune.exceptions import AITuneUserInputError
from aitune.torch.config import AITuneConfig
from aitune.torch.config import config as global_config
from aitune.torch.dynamic_shapes import DynamicShapes
from aitune.torch.module.forward_signature import ForwardInputPath, ForwardSignature
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.module.sample_store import Sample as Sample
from aitune.torch.module.sample_store import SampleStore
from aitune.torch.utils.path_utils import sanitize_filename

INPUT_METADATA_PREFIX = "input"
OUTPUT_METADATA_PREFIX = "output"

logger = logging.getLogger(__name__)


class RecordingModule:
    """Module that records samples for tuning."""

    def __init__(
        self,
        module: torch.nn.Module,
        name: str,
        config: AITuneConfig | None = None,
        dynamic_shapes: DynamicShapes | None = None,
    ) -> None:
        """Initialize BaseModule.

        Args:
            module: module to be tuned.
            name: name of the module.
            config: Configuration for the module, if not provided, global config is used.
            dynamic_shapes: Explicit input shape definitions.
        """
        super().__init__()
        if not isinstance(module, torch.nn.Module):
            raise AITuneUserInputError("Only torch modules are supported.")

        self._module = module
        self._name = name
        self._config = config if config is not None else global_config
        self._forward_call = module.__call__
        self._forward_signature = ForwardSignature.from_callable(module.forward)
        self._dynamic_shapes = dynamic_shapes or {}

        self._samples: dict[SampleMetadata, SampleStore] = {}
        # make temp directory to store samples, it has to be a field so that is not prematurely removed
        tempdir_prefix = sanitize_filename(self._name)
        self._temp_dir = tempfile.TemporaryDirectory(prefix=f"{tempdir_prefix}_")
        self._samples_dir = Path(self._temp_dir.name)
        self._graph_specs: OrderedDict[SampleMetadata, GraphSpec] = OrderedDict()
        self._graphs_counter = itertools.count()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Record a sample and run the module.

        Before calling forward, we need to make input sample metadata and a copy of args and kwargs, since the
        forward call may have side effects on the inputs (like KV cache in LLM models).
        """
        logger.debug("Calling recording %s module.", self._name)
        forward_inputs = self._forward_signature.normalize(args, kwargs)
        inputs_metadata = SampleMetadata.from_inputs(
            forward_inputs.arguments,
            strict=self._config.strict_mode,
        )
        dynamic_shapes = {}
        if self._dynamic_shapes:
            dynamic_shapes = self._resolve_and_validate_dynamic_shapes(inputs_metadata)
        sample_args, sample_kwargs = self._copy_inputs(forward_inputs.args, forward_inputs.kwargs)
        outputs = self._forward_call(*args, **kwargs)
        outputs_metadata = SampleMetadata.from_outputs(outputs, strict=self._config.strict_mode)

        self.record_sample((sample_args, sample_kwargs), inputs_metadata, outputs_metadata, dynamic_shapes)

        return outputs

    @property
    def device(self) -> torch.device:
        """Get the device of the module."""
        return next(self._module.parameters()).device

    @property
    def forward_signature(self) -> ForwardSignature:
        """Get the forward signature used to normalize inputs."""
        return self._forward_signature

    def record_sample(
        self, inputs, inputs_metadata, outputs_metadata, dynamic_shapes: DynamicShapes | None = None
    ) -> None:
        """Record a sample from the module."""
        if inputs_metadata in self._graph_specs:
            # graphs share same hash but can have different min, max seen shapes
            graph_spec = self._graph_specs[inputs_metadata]
            graph_spec.update_shapes_seen(inputs_metadata, outputs_metadata)
        else:
            # create a new graph spec for the sample metadata
            graph_name = f"{next(self._graphs_counter)}"
            if inputs_metadata.llm_phase:
                graph_name += f" (LLM phase {inputs_metadata.llm_phase})"
            self._graph_specs[inputs_metadata] = GraphSpec(
                name=graph_name,
                input_spec=inputs_metadata,
                output_spec=outputs_metadata,
                forward_signature=self._forward_signature,
                dynamic_shapes=dynamic_shapes or {},
            )
            self._samples[inputs_metadata] = SampleStore.create(
                self._samples_dir,
                f"graph-{graph_name}",
                owner=self._temp_dir,
            )

        if len(self._samples[inputs_metadata]) < self._config.max_num_samples_stored:
            self._samples[inputs_metadata].append(inputs)

    @property
    def is_ready_for_optimization(self) -> bool:
        """Check if the module is ready for tuning.

        All graphs would have proper amount of samples.
        """
        return all(len(samples) >= self._config.min_num_samples for samples in self._samples.values())

    @property
    def graph_specs(self) -> list[GraphSpec]:
        """Get the graph specs."""
        return list(self._graph_specs.values())

    def samples_for_graph_spec(self, graph_spec: GraphSpec) -> SampleStore:
        """Get the samples."""
        return self._samples[graph_spec.input_spec]

    def validate_dynamic_shape_paths_recorded(self) -> None:
        """Validate that every configured input path matched a tensor during recording."""
        matched_paths = {path for graph_spec in self._graph_specs.values() for path in graph_spec.dynamic_shapes}
        unmatched_paths = self._dynamic_shapes.keys() - matched_paths
        if unmatched_paths:
            raise AITuneUserInputError(
                f"Dynamic shape paths for module {self._name!r} did not match any recorded tensor inputs: "
                f"{sorted(unmatched_paths, key=repr)!r}."
            )

    def _resolve_and_validate_dynamic_shapes(self, inputs_metadata: SampleMetadata) -> DynamicShapes:
        """Match explicit shape definitions to tensors in a recorded sample.

        The returned mapping is a validated, graph-specific subset of ``self._dynamic_shapes`` containing only
        definitions whose paths resolve to tensor inputs in ``inputs_metadata``. Unconfigured tensor inputs remain
        represented by their ``TensorSpec`` and continue using shapes inferred from recorded samples.
        """
        resolved: DynamicShapes = {}
        dimension_sizes_by_name: dict[str, int] = {}
        for locator, tensor_spec in inputs_metadata.tensor_data:
            path = cast(ForwardInputPath, locator.path)
            shape_definition = self._dynamic_shapes.get(path)
            # Inputs without an explicit definition keep their inferred shapes.
            if shape_definition is None:
                continue

            recorded_shape = tensor_spec.min_shape
            # The shape definition rank must match the recorded tensor rank.
            if len(shape_definition) != len(recorded_shape):
                raise AITuneUserInputError(
                    f"Dynamic shape for module {self._name!r}, input {path!r} expects rank {len(shape_definition)}, "
                    f"got shape {recorded_shape!r}."
                )

            for axis, (dimension, recorded_size) in enumerate(zip(shape_definition, recorded_shape, strict=True)):
                # Integer dimensions are static and must match exactly.
                if isinstance(dimension, int):
                    if recorded_size != dimension:
                        raise AITuneUserInputError(
                            f"Dynamic shape for module {self._name!r}, input {path!r}, axis {axis} expects "
                            f"size {dimension}, got {recorded_size} from shape {recorded_shape!r}."
                        )
                    continue

                # Dynamic dimensions must contain the recorded size.
                if not dimension.min <= recorded_size <= dimension.max:
                    raise AITuneUserInputError(
                        f"Dynamic shape for module {self._name!r}, input {path!r}, axis {axis} expects "
                        f"{dimension.name!r} between {dimension.min} and {dimension.max}, got {recorded_size} "
                        f"from shape {recorded_shape!r}."
                    )

                # Track shared dimensions by name and require their sizes to match within this sample.
                if dimension.name not in dimension_sizes_by_name:
                    dimension_sizes_by_name[dimension.name] = recorded_size
                else:
                    expected_size = dimension_sizes_by_name[dimension.name]
                    if expected_size != recorded_size:
                        raise AITuneUserInputError(
                            f"Dynamic dimension {dimension.name!r} for module {self._name!r} must have the same size "
                            f"across inputs; input {path!r}, axis {axis} expected {expected_size}, got {recorded_size} "
                            f"from shape {recorded_shape!r}."
                        )

            resolved[path] = shape_definition

        return resolved

    @staticmethod
    def _copy_inputs(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """Create deep copies of args and kwargs.

        Args:
            args: Positional arguments tuple.
            kwargs: Keyword arguments dict.

        Returns:
            Tuple containing deep copies of args and kwargs.

        Raises:
            RuntimeError: If cannot copy model inputs with tip to fix it
        """
        try:
            return deepcopy(args), deepcopy(kwargs)
        except RuntimeError as e:
            # raise helpful error message
            raise RuntimeError(
                "Cannot copy model inputs. Model is not in no_grad mode. "
                "Tip: you can use \n---\nwith torch.no_grad():\n    model()\n---\ncontext manager to fix this."
            ) from e
