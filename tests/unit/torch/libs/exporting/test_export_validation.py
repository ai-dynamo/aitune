# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from aitune.torch.libs.exporting.export_validation import preflight_export, validate_export_profiles
from tests.utilities.helpers import make_recorded_graph_spec, update_input_spec


class Model(torch.nn.Module):
    def forward(self, x):
        return x + 1


def _export_and_spec(export_max=4, profile_max=4):
    model = Model()
    sample = ((torch.ones(2, 3),), {})
    exported = torch.export.export(
        model, sample[0], dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=export_max)}}
    )
    spec = make_recorded_graph_spec(model, sample)
    update_input_spec(spec, ((torch.ones(1, 3),), {}))
    update_input_spec(spec, ((torch.ones(profile_max, 3),), {}))
    return exported, spec


def test_export_preflight_and_profiles_do_not_rewrite_state():
    exported, spec = _export_and_spec()
    code, signature = exported.graph_module.code, str(exported.graph_signature)
    preflight_export(exported)
    validate_export_profiles(exported, spec)
    assert exported.graph_module.code == code
    assert str(exported.graph_signature) == signature


def test_profiles_reject_narrowed_export_without_clamping():
    exported, spec = _export_and_spec(export_max=3)
    with pytest.raises(ValueError, match="ranges were not clamped"):
        validate_export_profiles(exported, spec)
    assert spec.input_spec.tensor_data[0][1].max_shape[0] == 4


def test_profiles_reject_static_axis_mismatch():
    exported, spec = _export_and_spec()
    update_input_spec(spec, ((torch.ones(4, 5),), {}))
    with pytest.raises(ValueError, match="axis 1"):
        validate_export_profiles(exported, spec)


def test_preflight_reports_missing_lifted_state_without_repair(mocker):
    exported = mocker.Mock()
    error = RuntimeError("lifted_tensor_0 references nonexistent attribute")
    exported.module.side_effect = error
    with pytest.raises(RuntimeError, match="no tensors were relocated") as caught:
        preflight_export(exported)
    assert caught.value.__cause__ is error
    exported.validate.assert_called_once()
    exported.module.assert_called_once()
