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
"""Inplace wrap modules."""

import itertools
import logging
import tempfile
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import torch

from aitune.exceptions import AITuneUserInputError
from aitune.torch.config import AITuneConfig
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata

INPUT_METADATA_PREFIX = "input"
OUTPUT_METADATA_PREFIX = "output"

logger = logging.getLogger(__name__)

Sample = tuple[tuple[Any], dict]


class RecordingModule:
    """Module that records samples for tunning."""

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
            config: config for the module. If None, default config is used.
            device: Device on which tuned module has to be executed.
        """
        super().__init__()
        if not isinstance(module, torch.nn.Module):
            raise AITuneUserInputError("Only torch modules are supported.")

        self._module = module
        self._name = name
        self._config = config if config is not None else AITuneConfig()
        self._forward_call = module.__call__

        self._samples = defaultdict(list)
        self._total_num_samples = 0
        # make temp directory to store samples, it has to be a field so that is not prematurely removed
        self._temp_dir = tempfile.TemporaryDirectory(prefix=f"{self._name}_")
        self._samples_dir = Path(self._temp_dir.name)
        self._graph_specs: OrderedDict[SampleMetadata, GraphSpec] = OrderedDict()
        self._graphs_counter = itertools.count()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Record a sample and run the module."""
        logger.debug("Calling recording %s module.", self._name)
        output = self._forward_call(*args, **kwargs)

        inputs: Sample = (args, kwargs)
        self.record_sample(inputs, output)

        return output

    @property
    def device(self) -> torch.device:
        """Get the device of the module."""
        return next(self._module.parameters()).device

    def record_sample(self, inputs, outputs) -> None:
        """Record a sample from the module."""
        inputs_metadata = SampleMetadata.from_sample(inputs, prefix=INPUT_METADATA_PREFIX)
        outputs_metadata = SampleMetadata.from_sample(outputs, prefix=OUTPUT_METADATA_PREFIX)

        if inputs_metadata in self._graph_specs:
            # graphs share same hash but can have different min, max seen shapes
            self._graph_specs[inputs_metadata].update_shapes_seen(inputs_metadata, outputs_metadata)
        else:
            # create a new graph spec for the sample metadata
            graph_name = f"{next(self._graphs_counter)}"
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
            result.append(torch.load(path, weights_only=True))
        return result
