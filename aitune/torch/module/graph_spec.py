# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contains GraphSpec which represents a graph specification."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from aitune.torch.dynamic_shapes import BatchDim, DynamicShapes, ShapeDefinition
from aitune.torch.module.forward_signature import ForwardSignature
from aitune.torch.module.locator import Locator
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.module.tensor_spec import TensorSpec


@dataclass
class GraphSpec:
    """GraphSpec used to describe a computational graph.

    Each torch module has its own specification of input and output variables. The input specification i.e. args and kwargs
    of the torch modules `forward` function is represented by SampleMetadata. Those inputs can change computational
    graph. AITune treats each unique input specification as a separate graph which is tuned separately. This object
    represents such a computational graph with a name and input_spec information.

    Attributes:
        name (str) - symbolic name of the graph
        input_spec (SampleMetadata) - spec which describes the graph input before execution
        output_spec (SampleMetadata) - spec which describes the graph output
        forward_signature (ForwardSignature) - signature of the module's forward method
        dynamic_shapes (DynamicShapes) - explicit input shape definitions
        post_input_spec (SampleMetadata) - spec which describes the graph input after execution
    """

    name: str
    input_spec: SampleMetadata
    output_spec: SampleMetadata
    forward_signature: ForwardSignature
    dynamic_shapes: DynamicShapes = field(default_factory=dict)
    post_input_spec: SampleMetadata | None = None

    def update_shapes_seen(
        self,
        inputs_metadata: SampleMetadata,
        outputs_metadata: SampleMetadata,
        post_inputs_metadata: SampleMetadata | None = None,
    ):
        """Update metadata observed before, during, and after another call."""
        self.input_spec.update_shapes_seen(inputs_metadata)
        if post_inputs_metadata is not None:
            if self.post_input_spec is None:
                self.post_input_spec = post_inputs_metadata
            else:
                self.post_input_spec.update_shapes_seen(post_inputs_metadata)
        self.output_spec.update_shapes_seen(outputs_metadata)

    def make_batch(self, args: tuple, kwargs: dict[str, Any], batch_size: int) -> tuple[tuple, dict[str, Any]]:
        """Return a normalized call resized to the specified batch size."""
        forward_inputs = self.forward_signature.normalize(args, kwargs)
        forward_inputs.arguments = self.input_spec.make_batch(forward_inputs.arguments, batch_size)
        return forward_inputs.args, forward_inputs.kwargs

    def update_max_batch_size(self, sample: tuple[tuple, dict], max_batch_size: int) -> None:
        """Update input metadata with the specified maximum batch size."""
        args, kwargs = sample
        forward_inputs = self.forward_signature.normalize(args, kwargs)
        self.input_spec.update_max_batch_size(forward_inputs.arguments, max_batch_size)

    def get_shape_definition(self, locator: Locator) -> ShapeDefinition | None:
        """Return the explicit shape definition for an input tensor, if configured."""
        return self.dynamic_shapes.get(locator.path)

    def get_effective_input_shapes(
        self, locator: Locator, tensor_spec: TensorSpec
    ) -> tuple[list[int], list[int], list[int]]:
        """Return the minimum, optimal, and maximum shapes to use for compilation.

        An explicit user definition takes precedence. Otherwise, use the recorded TensorSpec bounds with its maximum
        shape as the optimal shape.
        """
        definition = self.get_shape_definition(locator)
        if definition is None:
            return tensor_spec.min_shape[:], tensor_spec.max_shape[:], tensor_spec.max_shape[:]

        min_shape: list[int] = []
        opt_shape: list[int] = []
        max_shape: list[int] = []
        for dimension in definition:
            if isinstance(dimension, int):
                # Static dimension
                min_shape.append(dimension)
                opt_shape.append(dimension)
                max_shape.append(dimension)
            else:
                min_shape.append(dimension.min)
                opt_shape.append(dimension.max if dimension.opt is None else dimension.opt)
                max_shape.append(dimension.max)
        return min_shape, opt_shape, max_shape

    def _iter_batch_size_ranges(self, normalized: bool = False) -> Iterator[tuple[int, int]]:
        """Yield batch ranges from explicit definitions or inferred metadata."""
        for locator, tensor_spec in self.input_spec.tensor_data:
            definition = self.get_shape_definition(locator)
            if definition is not None:
                for dimension in definition:
                    if isinstance(dimension, BatchDim):
                        yield dimension.min, dimension.max
                continue

            for axis, multiplier in tensor_spec.get_batch_axis_multipliers().items():
                multiplier = multiplier if normalized else 1
                yield tensor_spec.min_shape[axis] // multiplier, tensor_spec.max_shape[axis] // multiplier

    def get_max_batch_size(self, normalized: bool = False) -> int:
        """Get max batch size from input spec.

        Args:
            normalized: Flag to normalize the batch size against the global batch size.
        """
        return max((max_size for _, max_size in self._iter_batch_size_ranges(normalized)), default=1)

    def get_min_batch_size(self) -> int | None:
        """Get min batch size from input spec."""
        return min((min_size for min_size, _ in self._iter_batch_size_ranges()), default=None)

    def to_dict(self) -> dict[str, Any]:
        """Convert the graph specification to a serializable dictionary."""
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "input_spec": self.input_spec,
            "output_spec": self.output_spec,
            "forward_signature": self.forward_signature.to_dict(),
            "dynamic_shapes": self.dynamic_shapes,
            "post_input_spec": self.post_input_spec,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "GraphSpec":
        """Create GraphSpec from dictionary."""
        if data.get("type") != GraphSpec.__name__:
            raise ValueError(f"Invalid dictionary format for {GraphSpec.__name__}")
        return GraphSpec(
            name=data["name"],
            input_spec=data["input_spec"],
            output_spec=data["output_spec"],
            forward_signature=ForwardSignature.from_dict(data["forward_signature"]),
            dynamic_shapes=data["dynamic_shapes"],
            post_input_spec=data.get("post_input_spec"),
        )

    def __str__(self) -> str:
        """Return string representation of GraphSpec."""
        return f"Name={self.name}\nInput_spec:\n{self.input_spec.describe()}Output_spec:\n{self.output_spec.describe()}"

    def __repr__(self) -> str:
        """Return representation of GraphSpec."""
        return self.__str__()
