# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check exported programs and repair missing tensor constant paths for AOT backends."""

import logging
from dataclasses import replace

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TorchExportContractError(RuntimeError):
    """Raised when an exported program cannot preserve its execution contract."""


def validate_exported_module(exported_program: torch.export.ExportedProgram) -> None:
    """Validate the program and check that it can become an executable module.

    Try a repair only if creating the module raises AttributeError and a tensor
    constant refers to a parent module that no longer exists. After the repair,
    validate the program and try creating the module again.

    Keep this workaround here because it changes Torch's private graph signature,
    which records the graph inputs and outputs. Exporters call this function after
    tracing succeeds.
    """
    try:
        exported_program.validate()
    except Exception as error:
        raise TorchExportContractError("Torch Export produced an invalid program.") from error
    try:
        exported_program.module(check_guards=False)
        return
    except AttributeError as error:
        if not _move_tensor_constants_to_root(exported_program):
            raise TorchExportContractError(
                "Torch Export produced a program that cannot be converted to an executable module."
            ) from error

    try:
        exported_program.validate()
        exported_program.module(check_guards=False)
    except Exception as error:
        raise TorchExportContractError(
            "Torch Export produced a program whose lifted tensor constants cannot be restored on the root GraphModule."
        ) from error


def _move_tensor_constants_to_root(exported_program: torch.export.ExportedProgram) -> bool:
    """Move tensor constants with missing parent modules onto the root module.

    Some exported graphs still refer to a tensor constant at a path such as
    ``module.self_attn.lifted_tensor_0`` after the corresponding nested module was
    removed from the ``GraphModule``. ``ExportedProgram.module()`` then fails
    because it cannot find the parent module. Moving the constant onto the root
    module removes the reference to the missing parent.
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
            logger.warning("Moved tensor constant from %s to %s before AOT compilation.", target, root_target)

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
    """Return whether all parent objects in the attribute path exist."""
    parent = graph_module
    for component in target.split(".")[:-1]:
        if not hasattr(parent, component):
            return False
        parent = getattr(parent, component)
    return True
