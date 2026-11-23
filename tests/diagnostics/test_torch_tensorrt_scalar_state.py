# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolate scalar cache-state handling in upstream Torch-TensorRT, without Transformers or AITune.

Run explicitly on a GPU host:
    pytest tests/diagnostics/test_torch_tensorrt_scalar_state.py -v

This diagnostic is outside the normal unit/functional CI suites until its outcomes are established.
Both an initially CPU-resident counter (lazy cache) and a CUDA counter (initialized cache) are tested.
"""

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from tests.utilities.helpers import requires_cuda


class ScalarStateModel(nn.Module):
    """Move a scalar token count to the input device and increment it in place."""

    def forward(self, value: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = length.to(value.device)
        length.add_(1)
        return value + length, length


@requires_cuda
@pytest.mark.parametrize("execution", ["aot", "jit"])
@pytest.mark.parametrize("counter_device", ["cpu", "cuda"])
def test_scalar_state_preserves_shape_and_updates(execution: str, counter_device: str, tmp_path: Path):
    torch_tensorrt = pytest.importorskip("torch_tensorrt")
    model = ScalarStateModel().eval()
    value = torch.ones(1, 4, device="cuda")
    sample_length = torch.tensor(0, dtype=torch.int64, device=counter_device)
    settings: dict[str, Any] = {"min_block_size": 1, "require_full_compilation": True, "disable_tf32": True}

    with torch.no_grad():
        if execution == "aot":
            exported = torch.export.export(model, (value, sample_length.clone()), strict=False)
            compiled = torch_tensorrt.dynamo.compile(exported, inputs=[value, sample_length.clone()], **settings)
        else:
            compiled = torch.compile(model, backend="torch_tensorrt", options=settings, dynamic=False)

        _check_updates(compiled, value, counter_device)
        if execution == "aot":
            # Check serialization independently after confirming the in-memory compiled graph.
            path = tmp_path / "scalar_state.pt2"
            program = torch_tensorrt.dynamo.export(compiled)
            torch.export.save(program, path)
            restored = torch_tensorrt.load(str(path)).module()
            _check_updates(restored, value, counter_device)


def _check_updates(model, value: torch.Tensor, counter_device: str) -> None:
    reference = ScalarStateModel().eval()
    actual_length = torch.tensor(0, dtype=torch.int64, device=counter_device)
    expected_length = actual_length.clone()
    for step in range(1, 4):
        actual_input, expected_input = actual_length, expected_length
        expected, expected_length = reference(value * step, expected_input)
        actual, actual_length = model(value * step, actual_input)
        assert actual_length.shape == expected_length.shape == torch.Size([]), f"step {step}"
        torch.testing.assert_close(actual_length, expected_length)
        torch.testing.assert_close(actual, expected)
        # Check caller-visible mutation as well as the returned state, including the original CPU counter.
        torch.testing.assert_close(actual_input, expected_input)
