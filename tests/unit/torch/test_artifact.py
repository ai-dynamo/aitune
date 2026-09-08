# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Torch artifact tensor contracts."""

import torch

from aitune.records import BoundedTensorSpec, DType
from aitune.torch.artifact import bounded_tensor_specs
from aitune.torch.module.forward_signature import ForwardSignature
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata


def _graph_spec() -> GraphSpec:
    def forward(tokens, mask):
        return tokens, mask

    signature = ForwardSignature.from_callable(forward)
    graph_spec = None
    for batch_size in (1, 4):
        tokens = torch.zeros(batch_size, 8, dtype=torch.int64)
        mask = torch.zeros(batch_size * 2, 8, dtype=torch.bool)
        arguments = signature.normalize((tokens, mask), {}).arguments
        inputs = SampleMetadata.from_inputs(arguments, batch_size=batch_size)
        outputs = SampleMetadata.from_outputs((tokens, mask), batch_size=batch_size)
        if graph_spec is None:
            graph_spec = GraphSpec("graph", inputs, outputs, signature)
        else:
            graph_spec.update_shapes_seen(inputs, outputs)
    return graph_spec


def test_bounded_tensor_specs_derive_bounds_dtype_and_batch_axis_from_graph_spec():
    specs = bounded_tensor_specs(_graph_spec(), "input")

    assert specs == (
        BoundedTensorSpec(
            name="input_tokens",
            dtype=DType.INT64,
            min_shape=(1, 8),
            max_shape=(4, 8),
            batch_axis=0,
        ),
        BoundedTensorSpec(
            name="input_mask",
            dtype=DType.BOOL,
            min_shape=(2, 8),
            max_shape=(8, 8),
            batch_axis=None,
        ),
    )


def test_bounded_tensor_specs_apply_backend_order_and_names():
    specs = bounded_tensor_specs(
        _graph_spec(),
        "output",
        metadata_indices=(1, 0),
        artifact_names=("OUTPUT__0", "OUTPUT__1"),
    )

    assert tuple(spec.name for spec in specs) == ("OUTPUT__0", "OUTPUT__1")
    assert tuple(spec.dtype for spec in specs) == (DType.BOOL, DType.INT64)
