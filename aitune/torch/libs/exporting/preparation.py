# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare flat tensor inputs shared by AITune exporters."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from aitune.torch.libs.exporting.dynamic_shapes import prepare_export_sample
from aitune.torch.libs.exporting.tensor_export_adapter import TensorExportAdapter
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import Sample
from aitune.torch.utils.module import move_tensors_to_device


@dataclass(frozen=True)
class TensorExportPreparation:
    """Tensor-only module adapter and example inputs prepared for export."""

    adapter: TensorExportAdapter
    flat_inputs: tuple[torch.Tensor, ...]


def prepare_tensor_export(
    module: nn.Module,
    sample: Sample,
    graph_spec: GraphSpec,
    *,
    device: torch.device | str | None = None,
) -> TensorExportPreparation:
    """Prepare a structured module call for a tensor-only exporter.

    Args:
        module: Module to prepare.
        sample: Representative positional and keyword inputs.
        graph_spec: Recorded input and output metadata.
        device: Optional destination for ordinary tensor leaves.

    Returns:
        The module adapter and its flat example inputs.
    """
    prepared_sample: Sample = prepare_export_sample(sample, graph_spec)
    if device is not None:
        prepared_sample = move_tensors_to_device(prepared_sample, device)

    forward_inputs = graph_spec.forward_signature.normalize(*prepared_sample)
    flat_inputs = graph_spec.input_spec.flatten_tensors(forward_inputs.arguments)
    adapter = TensorExportAdapter(module, prepared_sample, graph_spec)
    return TensorExportPreparation(adapter=adapter, flat_inputs=flat_inputs)
