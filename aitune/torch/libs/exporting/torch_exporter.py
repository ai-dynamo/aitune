# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export modules through the flat tensor boundary with Torch Export."""

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch._dynamo.exc import UserError, UserErrorType
from torch.fx.experimental.symbolic_shapes import ConstraintViolationError
from torch.utils._pytree import tree_map

from aitune.torch.libs.exporting.dynamic_shapes import (
    TorchDynamicShapes,
    build_dynamic_shapes,
    log_dynamic_shapes,
)
from aitune.torch.libs.exporting.preparation import prepare_tensor_export
from aitune.torch.libs.exporting.torch_export_compat import (
    TorchExportContractError as TorchExportContractError,
)
from aitune.torch.libs.exporting.torch_export_compat import validate_exported_module
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
        fallback_to_dynamic_hints: Retry rejected explicit constraints with bounded ``Dim.DYNAMIC`` hints.
    """

    def __init__(
        self,
        *,
        use_auto: bool = True,
        strict: bool | None = None,
        fallback_to_dynamic_hints: bool = False,
    ):
        """Initialize the dynamic-shape and tracing modes."""
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
        """Prepare a tensor-only example call and export the module."""
        dynamic_shapes = build_dynamic_shapes(graph_spec, use_auto=self.use_auto)

        strict_modes = (True, False) if self.strict is None else (self.strict,)
        strict_error: Exception | None = None
        for strict in strict_modes:
            dynamic_hints = _as_dynamic_hints(dynamic_shapes)
            can_fallback_to_hints = self.fallback_to_dynamic_hints and dynamic_hints is not dynamic_shapes
            shape_attempts = (dynamic_shapes, dynamic_hints) if can_fallback_to_hints else (dynamic_shapes,)
            for attempt, attempt_shapes in enumerate(shape_attempts):
                # A failed trace may mutate temporary Python state, so each attempt gets a fresh adapter.
                preparation = prepare_tensor_export(module, sample, graph_spec, device=device)
                log_dynamic_shapes(attempt_shapes)
                logger.info("Exporting model with torch.export.export (strict=%s).", strict)
                try:
                    with torch.no_grad():
                        exported_program = torch.export.export(
                            preparation.adapter,
                            preparation.flat_inputs,
                            kwargs=None,
                            dynamic_shapes=attempt_shapes,
                            strict=strict,
                        )
                except Exception as error:
                    if attempt == 0 and can_fallback_to_hints and _is_constraint_violation(error):
                        logger.warning(
                            "Explicit dynamic-shape constraints were rejected; retrying export with bounded "
                            "dynamic hints."
                        )
                        continue
                    if strict_error is not None:
                        raise error from strict_error
                    if self.strict is not None:
                        raise
                    strict_error = error
                    logger.warning("Strict Torch Export failed; retrying with non-strict tracing: %s", error)
                    break
                else:
                    validate_exported_module(exported_program)
                    return TorchExportResult(exported_program=exported_program, dynamic_shapes=attempt_shapes)

        raise AssertionError("Torch Export tracing modes must not be empty")


def _is_constraint_violation(error: Exception) -> bool:
    """Return whether Torch Export rejected the supplied shape constraints."""
    return isinstance(error, ConstraintViolationError) or (
        isinstance(error, UserError) and getattr(error, "error_type", None) == UserErrorType.CONSTRAINT_VIOLATION
    )


def _as_dynamic_hints(dynamic_shapes: TorchDynamicShapes | None) -> TorchDynamicShapes | None:
    """Replace explicit dimensions with bounded dynamic hints while preserving shared dimensions."""
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
