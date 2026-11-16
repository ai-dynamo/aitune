# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check requested TensorRT profiles against the program being compiled.

Torch Export can specialize an axis or infer narrower bounds than the workload recorded by AITune.
Passing the original wider profile to TensorRT would then claim shapes the program does not cover.
Reject that candidate before compilation rather than silently reducing the requested workload.
"""

from typing import cast

import torch
from sympy import Expr
from torch.export.graph_signature import InputKind
from torch.utils._sympy.value_ranges import bound_sympy

from aitune.torch.module.graph_spec import GraphSpec


def validate_export_profiles(exported: torch.export.ExportedProgram, graph_spec: GraphSpec) -> None:
    """Reject TRT profiles outside export bounds, without clamping requested shapes.

    Match exported USER_INPUT placeholders to recorded tensors in flat input order. Each input must
    retain tensor metadata and the expected rank. Requested min/opt/max sizes must be ordered and lie
    within each exported axis range; a specialized axis accepts only its fixed size.

    For example, a requested batch profile [1, 2, 4] cannot use an export bounded by [1, 3]. Raising
    keeps the missing batch-4 workload visible; changing the profile to [1, 2, 3] would hide it.
    Neither the program nor GraphSpec is rewritten. This checks axis bounds, not every symbolic
    relationship or numerical behavior; ordinary backend correctness checks remain necessary.

    Args:
        exported: Program whose shape constraints will be passed to compilation.
        graph_spec: Recorded tensor order and effective requested profiles, including user overrides.

    Raises:
        ValueError: If input counts, metadata, ranks, or requested axis bounds do not match the export.
    """
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
                bounds = bound_sympy(cast(Expr, dimension.node.expr), exported.range_constraints)
                lower, upper = bounds.lower, bounds.upper
            else:
                lower = upper = int(dimension)
            if not lower <= minimum[axis] <= optimum[axis] <= maximum[axis] <= upper:
                raise ValueError(
                    f"Export/profile mismatch for {locator}, axis {axis}: requested "
                    f"[{minimum[axis]}, {optimum[axis]}, {maximum[axis]}], exported [{lower}, {upper}]. "
                    "Re-export or split the graph to cover the requested workload; ranges were not clamped."
                )
