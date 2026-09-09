# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export modules through the flat tensor boundary with Torch Export."""

import logging
from dataclasses import dataclass, replace

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
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import Sample

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TorchExportResult:
    """Program and dynamic-shape rules produced by Torch Export."""

    exported_program: torch.export.ExportedProgram
    dynamic_shapes: TorchDynamicShapes | None


class TorchExportContractError(RuntimeError):
    """Raised when an exported program cannot preserve its execution contract."""


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
                    _preflight_unlift(exported_program)
                    return TorchExportResult(exported_program=exported_program, dynamic_shapes=attempt_shapes)

        raise AssertionError("Torch Export tracing modes must not be empty")


def _preflight_unlift(exported_program: torch.export.ExportedProgram) -> None:
    """Exercise the unlift operation required by AOT backends and repair nested tensor constants."""
    try:
        exported_program.validate()
    except Exception as error:
        raise TorchExportContractError("Torch Export produced an invalid program.") from error
    try:
        exported_program.module(check_guards=False)
        return
    except AttributeError as error:
        if not _rehome_lifted_tensor_constants(exported_program):
            raise TorchExportContractError("Torch Export produced a program that cannot be unlifted.") from error

    try:
        exported_program.validate()
        exported_program.module(check_guards=False)
    except Exception as error:
        raise TorchExportContractError(
            "Torch Export produced a program whose lifted tensor constants cannot be restored on the root GraphModule."
        ) from error


def _rehome_lifted_tensor_constants(exported_program: torch.export.ExportedProgram) -> bool:
    """Move nested lifted tensor constants to stable root attributes.

    Some exported adapter graphs retain a constant target such as
    ``module.self_attn.lifted_tensor_0`` after the corresponding nested module was
    removed from the owning ``GraphModule``. ``ExportedProgram.module()`` then
    fails while resolving the missing parent. Root targets do not depend on that
    transient module hierarchy.
    """
    signature = exported_program.graph_signature
    constants = exported_program.constants
    replacements = {}
    input_specs = []
    changed = False

    for input_spec in signature.input_specs:
        target = input_spec.target
        kind = getattr(input_spec.kind, "name", str(input_spec.kind))
        if kind != "CONSTANT_TENSOR" or not isinstance(target, str) or "." not in target:
            input_specs.append(input_spec)
            continue

        if target not in constants:
            input_specs.append(input_spec)
            continue

        if _has_nested_parent(exported_program.graph_module, target):
            input_specs.append(input_spec)
            continue

        root_target = replacements.get(target)
        if root_target is None:
            stem = "_aitune_lifted_tensor_" + target.replace(".", "_")
            root_target = stem
            suffix = 0
            while hasattr(exported_program.graph_module, root_target) or root_target in constants:
                suffix += 1
                root_target = f"{stem}_{suffix}"
            replacements[target] = root_target
            value = constants[target]
            setattr(exported_program.graph_module, root_target, value)
            constants[root_target] = value
            logger.warning("Re-homed lifted tensor constant %s to %s before AOT compilation.", target, root_target)

        input_specs.append(replace(input_spec, target=root_target))
        changed = True

    if not changed:
        return False

    repaired_signature = type(signature)(input_specs, signature.output_specs)
    exported_program._graph_signature = repaired_signature
    for target in replacements:
        if all(spec.target != target for spec in repaired_signature.input_specs):
            constants.pop(target, None)
    return True


def _has_nested_parent(graph_module: nn.Module, target: str) -> bool:
    """Return whether every parent component of an attribute target exists."""
    parent = graph_module
    for component in target.split(".")[:-1]:
        if not hasattr(parent, component):
            return False
        parent = getattr(parent, component)
    return True


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
