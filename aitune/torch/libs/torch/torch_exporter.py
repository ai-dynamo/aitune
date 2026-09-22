# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Torch Export functionality for ahead-of-time backends."""

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch._dynamo.exc import UserError, UserErrorType
from torch.fx.experimental.symbolic_shapes import ConstraintViolationError
from torch.utils._pytree import tree_map

from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import Sample
from aitune.torch.utils.module import move_tensors_to_device
from aitune.torch.utils.shapes import build_dynamic_shapes, log_dynamic_shapes, prepare_export_sample

logger = logging.getLogger(__name__)


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
        fallback_to_dynamic_hints: Retry constraint violations with bounded
            ``Dim.DYNAMIC`` hints instead of explicit dimensions.
    """

    def __init__(
        self,
        *,
        use_auto: bool = True,
        strict: bool = True,
        fallback_to_dynamic_hints: bool = False,
    ):
        """Initialize the exporter with its dynamic-shape and tracing modes."""
        self.use_auto = use_auto
        self.strict = strict
        self.fallback_to_dynamic_hints = fallback_to_dynamic_hints

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
            prepared_sample = move_tensors_to_device(prepared_sample, device)
        args, kwargs = prepared_sample
        dynamic_shapes = build_dynamic_shapes(prepared_sample, graph_spec, use_auto=self.use_auto)
        log_dynamic_shapes(dynamic_shapes)

        logger.info("Exporting model with torch.export.export.")
        with torch.no_grad():
            try:
                exported_program = torch.export.export(
                    module,
                    args,
                    kwargs=kwargs or None,
                    dynamic_shapes=dynamic_shapes,
                    strict=self.strict,
                )
            except (ConstraintViolationError, UserError) as error:
                if not self.fallback_to_dynamic_hints:
                    raise
                if isinstance(error, UserError) and error.error_type != UserErrorType.CONSTRAINT_VIOLATION:
                    raise
                dynamic_hints = _as_dynamic_hints(dynamic_shapes)
                if dynamic_hints is dynamic_shapes:
                    raise
                dynamic_shapes = dynamic_hints
                logger.warning(
                    "Explicit dynamic-shape constraints were rejected; retrying export with bounded dynamic hints."
                )
                log_dynamic_shapes(dynamic_shapes)
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


def _as_dynamic_hints(dynamic_shapes: dict[str, Any] | None) -> dict[str, Any] | None:
    """Relax explicit dimensions, returning the original structure when nothing changes."""
    if dynamic_shapes is None:
        return None

    hints = {}

    def replace(value):
        if not isinstance(value, torch.export.dynamic_shapes._Dim):
            return value
        if value not in hints:
            hints[value] = torch.export.Dim.DYNAMIC(min=value.min, max=value.max)
        return hints[value]

    dynamic_hints = tree_map(replace, dynamic_shapes)
    return dynamic_hints if hints else dynamic_shapes
