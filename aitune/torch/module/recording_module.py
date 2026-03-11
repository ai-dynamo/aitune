# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inplace wrap modules."""

import itertools
import logging
import tempfile
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from aitune.exceptions import AITuneUserInputError
from aitune.torch.config import AITuneConfig
from aitune.torch.config import config as global_config
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.utils.path_utils import sanitize_filename

INPUT_METADATA_PREFIX = "input"
OUTPUT_METADATA_PREFIX = "output"

logger = logging.getLogger(__name__)

Sample = tuple[tuple, dict]


class RecordingModule:
    """Module that records samples for tuning."""

    def __init__(
        self,
        module: torch.nn.Module,
        name: str,
        config: AITuneConfig | None = None,
    ) -> None:
        """Initialize BaseModule.

        Args:
            module: module to be tuned.
            name: name of the module.
            config: Configuration for the module, if not provided, global config is used.
        """
        super().__init__()
        if not isinstance(module, torch.nn.Module):
            raise AITuneUserInputError("Only torch modules are supported.")

        self._module = module
        self._name = name
        self._config = config if config is not None else global_config
        self._forward_call = module.__call__

        self._samples = defaultdict(list)
        self._total_num_samples = 0

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
        inputs_metadata = SampleMetadata.from_inputs(args, kwargs, strict=self._config.strict_mode)
        original_args, original_kwargs = self._copy_inputs(args, kwargs)
        outputs = self._forward_call(*args, **kwargs)
        outputs_metadata = SampleMetadata.from_outputs(outputs, strict=self._config.strict_mode)

        self.record_sample((original_args, original_kwargs), inputs_metadata, outputs_metadata)

        return outputs

    @property
    def device(self) -> torch.device:
        """Get the device of the module."""
        return next(self._module.parameters()).device

    def record_sample(self, inputs, inputs_metadata, outputs_metadata) -> None:
        """Record a sample from the module."""
        if inputs_metadata in self._graph_specs:
            # graphs share same hash but can have different min, max seen shapes
            self._graph_specs[inputs_metadata].update_shapes_seen(inputs_metadata, outputs_metadata)
        else:
            # create a new graph spec for the sample metadata
            graph_name = f"{next(self._graphs_counter)}"
            if inputs_metadata.llm_phase:
                graph_name += f" (LLM phase {inputs_metadata.llm_phase})"
            self._graph_specs[inputs_metadata] = GraphSpec(
                name=graph_name, input_spec=inputs_metadata, output_spec=outputs_metadata
            )

        if len(self._samples[inputs_metadata]) < self._config.max_num_samples_stored:
            sample_path = self._samples_dir / f"{self._total_num_samples}.pt"
            self._total_num_samples += 1
            torch.save(inputs, sample_path)
            self._samples[inputs_metadata].append(sample_path)

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

    def samples_for_graph_spec(self, graph_spec: GraphSpec) -> list[Sample]:
        """Get the samples."""
        result = []
        for path in self._samples[graph_spec.input_spec]:
            result.append(torch.load(path, weights_only=False))
        return result

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
                "Cannot copy model inputs. Model is not in inference mode. "
                "Tip: you can use \n---\nwith torch.no_grad():\n    model()\n---\ncontext manager to fix this."
            ) from e
