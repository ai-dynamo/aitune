# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export modules through the flat tensor boundary with Torch Export."""

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

from aitune.torch.libs.exporting.dynamic_shapes import (
    TorchDynamicShapes,
    build_dynamic_shapes,
    log_dynamic_shapes,
)
from aitune.torch.libs.exporting.preparation import prepare_tensor_export
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import Sample

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TorchExportResult:
    """Program and dynamic-shape rules produced by Torch Export."""

    exported_program: torch.export.ExportedProgram
    dynamic_shapes: TorchDynamicShapes | None


class TorchExporter:
    """Export a module through AITune's flat tensor interface.

    Args:
        use_auto: Let Torch Export infer constraints for non-batch dynamic dimensions.
        strict: Select strict or non-strict export. ``None`` tries strict export first and then falls back.
    """

    def __init__(self, *, use_auto: bool = True, strict: bool | None = None):
        """Initialize the dynamic-shape and tracing modes."""
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
        """Prepare a tensor-only example call and export the module."""
        dynamic_shapes = build_dynamic_shapes(graph_spec, use_auto=self.use_auto)
        log_dynamic_shapes(dynamic_shapes)

        strict_modes = (True, False) if self.strict is None else (self.strict,)
        strict_error: Exception | None = None
        for strict in strict_modes:
            # A failed trace may mutate temporary Python state, so each attempt gets a fresh adapter.
            preparation = prepare_tensor_export(module, sample, graph_spec, device=device)
            logger.info("Exporting model with torch.export.export (strict=%s).", strict)
            try:
                with torch.no_grad():
                    exported_program = torch.export.export(
                        preparation.adapter,
                        preparation.flat_inputs,
                        kwargs=None,
                        dynamic_shapes=dynamic_shapes,
                        strict=strict,
                    )
            except Exception as error:
                if strict_error is not None:
                    raise error from strict_error
                if self.strict is not None:
                    raise
                strict_error = error
                logger.warning("Strict Torch Export failed; retrying with non-strict tracing: %s", error)
            else:
                return TorchExportResult(exported_program=exported_program, dynamic_shapes=dynamic_shapes)

        raise AssertionError("Torch Export tracing modes must not be empty")
