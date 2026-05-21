# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared torch.compile helpers."""

from typing import Literal

from aitune.torch.module.graph_spec import GraphSpec

TorchCompileMode = Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]


def resolve_compile_dynamic(config_dynamic: bool | None, graph_spec: GraphSpec) -> bool | None:
    """Resolve effective dynamic compilation setting without mutating backend config.

    Args:
        config_dynamic: Explicit dynamic setting from backend config.
        graph_spec: Graph specification used for dynamic-axis detection.

    Returns:
        Effective value to pass to ``torch.compile(dynamic=...)``.
    """
    if config_dynamic is not None:
        return config_dynamic
    if graph_spec.input_spec.detected_dynamic_axis():
        return True
    return None
