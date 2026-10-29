# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for repairing Torch Export programs before AOT compilation."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from aitune.torch.libs.exporting.torch_export_compat import TorchExportContractError, validate_exported_module


def test_validation_moves_a_nested_tensor_constant_to_root(mocker):
    @dataclass(frozen=True)
    class InputSpec:
        kind: object
        target: str

    class Signature:
        def __init__(self, input_specs, output_specs):
            self.input_specs = input_specs
            self.output_specs = output_specs

    lifted = torch.ones(2)
    exported_program = mocker.Mock()
    exported_program.graph_module = torch.nn.Module()
    exported_program.constants = {"module.self_attn.lifted_tensor_0": lifted}
    exported_program.graph_signature = Signature(
        [InputSpec(SimpleNamespace(name="CONSTANT_TENSOR"), "module.self_attn.lifted_tensor_0")],
        [],
    )
    exported_program.module.side_effect = [AttributeError("self_attn"), mocker.sentinel.module]

    validate_exported_module(exported_program)

    repaired = exported_program._graph_signature
    assert repaired.input_specs[0].target.startswith("_aitune_lifted_tensor_")
    assert "." not in repaired.input_specs[0].target
    assert getattr(exported_program.graph_module, repaired.input_specs[0].target) is lifted
    assert exported_program.constants[repaired.input_specs[0].target] is lifted
    assert "module.self_attn.lifted_tensor_0" not in exported_program.constants
    assert exported_program.module.call_count == 2


def test_validation_reports_module_creation_failure(mocker):
    exported_program = mocker.Mock()
    exported_program.graph_signature.input_specs = []
    exported_program.module.side_effect = AttributeError("missing state")

    with pytest.raises(TorchExportContractError, match="cannot be converted to an executable module"):
        validate_exported_module(exported_program)
