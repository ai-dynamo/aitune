# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Torch Export functionality for ahead-of-time backends."""

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map_only

from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import Sample
from aitune.torch.utils.shapes import build_dynamic_shapes, log_dynamic_shapes, prepare_export_sample

logger = logging.getLogger(__name__)


def _move_to_device(value, device):
    return tree_map_only(
        torch.Tensor,
        lambda tensor: tensor.to(device),
        value,
    )


@dataclass(frozen=True)
class TorchExportResult:
    """Result of exporting a module and the exact inputs used for capture."""

    exported_program: torch.export.ExportedProgram
    sample: Sample
    dynamic_shapes: dict[str, Any] | None


class TorchExporter:
    """Prepare and export a module through ``torch.export.export``.

    Args:
        use_auto: Use automatic constraints for non-batch dynamic dimensions. Set to
            ``False`` when a consumer requires explicit minimum and maximum bounds.
        strict: Whether Torch Export should use strict tracing.
    """

    def __init__(self, *, use_auto: bool = True, strict: bool = True):
        """Initialize the exporter with its dynamic-shape and tracing modes."""
        self.use_auto = use_auto
        self.strict = strict

    def export(
        self,
        module: nn.Module,
        sample: Sample,
        graph_spec: GraphSpec,
        *,
        device: torch.device | str | None = None,
    ) -> TorchExportResult:
        """Prepare the sample, build matching shape constraints, and export the module.

        Args:
            module: Module to export.
            sample: Representative positional and keyword inputs.
            graph_spec: Recorded input and output metadata.
            device: Optional destination for every tensor leaf in the prepared sample.

        Returns:
            The exported program and the exact sample and constraints used for capture.
        """
        prepared_sample = prepare_export_sample(sample, graph_spec)
        if device is not None:
            prepared_sample = _move_to_device(prepared_sample, device)
        args, kwargs = prepared_sample
        dynamic_shapes = build_dynamic_shapes(prepared_sample, graph_spec, use_auto=self.use_auto)
        log_dynamic_shapes(dynamic_shapes)

        logger.info("Exporting model with torch.export.export.")
        with torch.no_grad():
            exported_program = torch.export.export(
                module,
                args,
                kwargs=kwargs or None,
                dynamic_shapes=dynamic_shapes,
                strict=self.strict,
            )

        return TorchExportResult(
            exported_program=exported_program,
            sample=prepared_sample,
            dynamic_shapes=dynamic_shapes,
        )


__all__ = ["TorchExporter", "TorchExportResult"]
