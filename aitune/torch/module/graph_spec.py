# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
"""Contains GraphSpec which represents a graph specification."""

from dataclasses import asdict, dataclass
from typing import Any

from aitune.torch.module.sample_metadata import SampleMetadata


@dataclass
class GraphSpec:
    """GraphSpec used to describe a computational graph.

    Each torch module has its own specification of input and output variables. The input specification i.e. args and kwargs
    of the torch modules `forward` function is represented by SampleMetadata. Those inputs can change computational
    graph. AITune treats each unique input specification as a separate graph which is tuned separately. This object
    represents such a computational graph with a name and input_spec information.

    Attributes:
        name (str) - symbolic name of the graph
        input_spec (SampleMetadata) - spec which describes the graph input
        output_spec (SampleMetadata) - spec which describes the graph output
    """

    name: str
    input_spec: SampleMetadata
    output_spec: SampleMetadata

    def update_shapes_seen(self, inputs_metadata: SampleMetadata, outputs_metadata: SampleMetadata):
        """Update input spec with other input spec."""
        self.input_spec.update_shapes_seen(inputs_metadata)
        self.output_spec.update_shapes_seen(outputs_metadata)

    def get_max_batch_size(self) -> int:
        """Get max batch size from input spec."""
        max_batch_size = 1
        for tensor_spec in self.input_spec.tensor_specs:
            for axis in tensor_spec.get_batch_axis_multipliers().keys():
                max_batch_size = max(max_batch_size, tensor_spec.max_shape[axis])
        return max_batch_size

    def get_min_batch_size(self) -> int | None:
        """Get min batch size from input spec."""
        min_batch_size = float("inf")
        for tensor_spec in self.input_spec.tensor_specs:
            for axis in tensor_spec.get_batch_axis_multipliers().keys():
                min_batch_size = min(min_batch_size, tensor_spec.min_shape[axis])
        if min_batch_size == float("inf"):
            return None
        return int(min_batch_size)

    def to_dict(self) -> dict[str, Any]:
        """Convert TensorSpec to a serializable dictionary."""
        return {"type": self.__class__.__name__} | asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "GraphSpec":
        """Create GraphSpec from dictionary."""
        if data.get("type") != GraphSpec.__name__:
            raise ValueError(f"Invalid dictionary format for {GraphSpec.__name__}")
        del data["type"]
        return GraphSpec(**data)

    def __str__(self) -> str:
        """Return string representation of GraphSpec."""
        return f"Name={self.name}\nInput_spec:\n{self.input_spec.describe()}Output_spec:\n{self.output_spec.describe()}"

    def __repr__(self) -> str:
        """Return representation of GraphSpec."""
        return self.__str__()
