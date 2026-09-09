# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check TensorRT optimization profiles against Torch Export shape constraints."""

import torch
from torch.export.graph_signature import InputKind
from torch.utils._sympy.value_ranges import bound_sympy

from aitune.torch.module.graph_spec import GraphSpec


def validate_export_profiles(exported: torch.export.ExportedProgram, graph_spec: GraphSpec) -> None:
    """Reject TRT profiles outside export bounds, without clamping requested shapes."""
    placeholders = {node.name: node for node in exported.graph.nodes if node.op == "placeholder"}
    user_inputs = [spec for spec in exported.graph_signature.input_specs if spec.kind == InputKind.USER_INPUT]
    tensors = graph_spec.input_spec.tensor_data
    if len(user_inputs) != len(tensors):
        raise ValueError("Exported user-input count does not match the flat AITune tensor contract")
    for spec, (locator, tensor_spec) in zip(user_inputs, tensors, strict=True):
        value = placeholders[spec.arg.name].meta.get("val")
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Exported tensor metadata is missing for {locator}")
        minimum, optimum, maximum = graph_spec.get_effective_input_shapes(locator, tensor_spec)
        if not (len(value.shape) == len(minimum) == len(optimum) == len(maximum)):
            raise ValueError(f"Export/profile rank mismatch for {locator}")
        for axis, dimension in enumerate(value.shape):
            if isinstance(dimension, torch.SymInt):
                bounds = bound_sympy(dimension.node.expr, exported.range_constraints)
                lower, upper = bounds.lower, bounds.upper
            else:
                lower = upper = int(dimension)
            if not lower <= minimum[axis] <= optimum[axis] <= maximum[axis] <= upper:
                raise ValueError(
                    f"Export/profile mismatch for {locator}, axis {axis}: requested "
                    f"[{minimum[axis]}, {optimum[axis]}, {maximum[axis]}], exported [{lower}, {upper}]. "
                    "Re-export or split the graph to cover the requested workload; ranges were not clamped."
                )
