# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helper functions and classes for testing."""

from typing import Any

import pytest

from aitune.torch.module.forward_signature import ForwardSignature
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.utils.cuda_utils import is_available as is_cuda_available

requires_cuda = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available")


def make_input_metadata(
    forward_signature: ForwardSignature,
    sample: tuple[tuple, dict[str, Any]],
    *,
    strict: bool = False,
    batch_size: int | None = None,
) -> SampleMetadata:
    """Create input metadata using the provided forward signature."""
    args, kwargs = sample
    forward_inputs = forward_signature.normalize(args, kwargs)
    return SampleMetadata.from_inputs(forward_inputs.arguments, strict=strict, batch_size=batch_size)


def make_graph_spec(
    forward: Any,
    sample: tuple[tuple, dict[str, Any]],
    output: Any = None,
    *,
    name: str = "test_graph",
    strict: bool = False,
    batch_size: int | None = None,
) -> GraphSpec:
    """Create a graph specification for a forward call."""
    forward_signature = ForwardSignature.from_callable(forward)
    return GraphSpec(
        name=name,
        input_spec=make_input_metadata(forward_signature, sample, strict=strict, batch_size=batch_size),
        output_spec=SampleMetadata.from_outputs(output, strict=strict, batch_size=batch_size),
        forward_signature=forward_signature,
    )


def update_input_spec(
    graph_spec: GraphSpec,
    sample: tuple[tuple, dict[str, Any]],
    *,
    strict: bool = False,
    batch_size: int | None = None,
) -> None:
    """Update graph input metadata from another forward call."""
    graph_spec.input_spec.update_shapes_seen(
        make_input_metadata(graph_spec.forward_signature, sample, strict=strict, batch_size=batch_size)
    )


class TestSink:
    """Sink for capturing output from PatchedModule.print_hierarchy.

    This class provides a simple interface for capturing text output
    that can be used to test print statements and other text-based
    output in tests.
    """

    def __init__(self):
        """Initialize the TestSink with an empty output list."""
        self.output = []

    def write(self, text):
        """Write text to the sink.

        Args:
            text (str): The text to append to the output list.
        """
        self.output.append(text)
