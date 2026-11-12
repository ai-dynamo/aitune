# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Call signatures, structured inputs, and structured outputs across backends."""

from collections.abc import Callable
from dataclasses import replace
from functools import partial, wraps

import pytest
import torch
from torch import nn

import aitune.torch as ait
from tests.functional_ci.backends import BACKENDS, BackendCase, with_backends
from tests.functional_ci.lifecycle import assert_matches_eager, pinned_strategy, selected_backend
from tests.functional_ci.models import IN_FEATURES, make_input, make_mlp


def require_keyword_arguments(forward):
    @wraps(forward)
    def wrapped(self, *args, **kwargs):
        if args:
            raise TypeError("This model requires keyword arguments")
        return forward(self, **kwargs)

    return wrapped


class StructuredModel(nn.Module):
    """Model with nested dict and list inputs, non-tensor and optional arguments, and a nested dict/tuple output."""

    def __init__(self, mlp: nn.Module):
        """Wrap the toy MLP."""
        super().__init__()
        self.mlp = mlp

    def forward(
        self,
        x,
        extras: dict[str, torch.Tensor],
        residuals: list[torch.Tensor],
        scale: float = 1.0,
        mask: torch.Tensor | None = None,
    ):
        """Return the scaled MLP output and two row statistics."""
        hidden = self.mlp(x + extras["bias"] + residuals[0] - residuals[1]) * scale
        if mask is not None:
            hidden = hidden * mask
        return {
            "out": hidden,
            "stats": (hidden.sum(-1), hidden.amax(-1)),
            "metadata": {"mask": None, "scale": scale, "enabled": True},
        }


class NestedInputModel(nn.Module):
    """Model with only tensor inputs, shaped like the Flux transformer inputs in legacy test 044."""

    def __init__(self, mlp: nn.Module):
        """Wrap the toy MLP."""
        super().__init__()
        self.mlp = mlp

    def forward(
        self,
        x,
        extras: dict[str, torch.Tensor],
        residuals: list[torch.Tensor],
        positions: torch.Tensor,
        timestep: torch.Tensor,
    ):
        """Return the MLP output for the shifted input, scaled per row by `timestep`."""
        hidden = x + extras["bias"] + residuals[0] - residuals[1] + positions
        return self.mlp(hidden * timestep[:, None])


class KeywordOnlyModel(nn.Module):
    """Keep a keyword-enforcing decorator active during recording and backend execution."""

    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp

    @require_keyword_arguments
    def forward(self, x, *, scale=2.0, mask=None):
        output = self.mlp(x) * scale
        return output if mask is None else output * mask


def structured_pipeline(model: Callable, bias, residuals):
    # Every tensor in the sample is batched, so the bias and residuals are repeated per row.
    return lambda x: model(
        x,
        {"bias": bias.repeat(x.shape[0], 1)},
        [residual.repeat(x.shape[0], 1) for residual in residuals],
        scale=2.0,
        mask=None,
    )


def nested_tensor_pipeline(model: Callable, bias, residuals):
    # `positions` keeps one shape at every batch size, like `img_ids` in 044; the other tensors are batched.
    return lambda x: model(
        x,
        extras={"bias": bias.repeat(x.shape[0], 1)},
        residuals=[residual.repeat(x.shape[0], 1) for residual in residuals],
        positions=bias.flip(0),
        timestep=torch.linspace(0.5, 1.5, x.shape[0], device=x.device),
    )


def keyword_call(model, value):
    return model(x=value, scale=2.0, mask=None)


@pytest.mark.matrix(
    "aot_tuning.models.non-tensor-inputs",
    "aot_tuning.models.nested-inputs",
    "aot_tuning.models.structured-outputs",
    "modes.aot-save-load",
)
@with_backends("base")
def test_structured_io(case: BackendCase, device: torch.device, tmp_path):
    """Nested inputs and mixed tensor/constant outputs survive tuning and checkpoint reload."""
    backend = case.factory()
    # Row sums in `stats` accumulate the per-element error over `OUT_FEATURES` values, so widen `atol`.
    tolerance = replace(case.tolerance, atol=5 * case.tolerance.atol)
    bias = torch.linspace(-1, 1, IN_FEATURES, device=device)
    residuals = [bias.flip(0), 0.5 * bias]
    reference = StructuredModel(make_mlp(device))
    tuned = ait.Module(StructuredModel(make_mlp(device)), "structured", strategy=pinned_strategy(backend))

    ait.tune(
        structured_pipeline(tuned, bias, residuals),
        list(make_input(device, batch_size=8).unbind(0)),
        batch_sizes=[1, 4],
        device=device,
    )

    selected_backend(tuned, type(backend))
    assert_matches_eager(
        structured_pipeline(tuned, bias, residuals),
        structured_pipeline(reference, bias, residuals),
        device,
        tolerance,
        batch_sizes=(1, 3, 4),
    )

    checkpoint = tmp_path / "structured.ait"
    ait.save(tuned, checkpoint)
    tuned.deactivate()
    restored = ait.load(StructuredModel(make_mlp(device, seed=1)), checkpoint)
    assert isinstance(restored, ait.Module)
    selected_backend(restored, type(backend))
    assert_matches_eager(
        structured_pipeline(restored, bias, residuals),
        structured_pipeline(reference, bias, residuals),
        device,
        tolerance,
        batch_sizes=(1, 3, 4),
    )
    with torch.no_grad():
        output = structured_pipeline(restored, bias, residuals)(make_input(device))
    assert isinstance(output["stats"], tuple)
    assert output["metadata"] == {"mask": None, "scale": 2.0, "enabled": True}
    assert output["metadata"]["mask"] is None
    assert type(output["metadata"]["scale"]) is float
    assert output["metadata"]["enabled"] is True


@pytest.mark.matrix("aot_tuning.models.nested-inputs/torch-tensorrt-aot")
@pytest.mark.parametrize("batch_sizes", [[2], [2, 4]], ids=["static", "dynamic"])
def test_nested_tensor_inputs(batch_sizes: list[int], device: torch.device):
    """Torch-TensorRT AOT tunes nested tensor inputs passed as kwargs at static and dynamic batch, as in legacy 044.

    The inputs are a dict, a list, an unbatched tensor, and a 1-D batched scalar. `test_structured_io` covers only a
    nested dict and list, together with non-tensor arguments.
    """
    case = BACKENDS["torch-tensorrt-aot"]
    backend = case.factory()
    bias = torch.linspace(-1, 1, IN_FEATURES, device=device)
    residuals = [bias.flip(0), 0.5 * bias]
    reference = NestedInputModel(make_mlp(device))
    tuned = ait.Module(NestedInputModel(make_mlp(device)), "nested", strategy=pinned_strategy(backend))

    ait.tune(
        nested_tensor_pipeline(tuned, bias, residuals),
        list(make_input(device, batch_size=8).unbind(0)),
        batch_sizes=batch_sizes,
        device=device,
    )

    selected_backend(tuned, type(backend))
    check_sizes = range(batch_sizes[0], batch_sizes[-1] + 1)
    assert_matches_eager(
        nested_tensor_pipeline(tuned, bias, residuals),
        nested_tensor_pipeline(reference, bias, residuals),
        device,
        case.tolerance,
        batch_sizes=check_sizes,
    )


@pytest.mark.matrix("aot_tuning.models.keyword-call-layout", "modes.aot-save-load")
@with_backends("base")
def test_keyword_call_layout_lifecycle(case: BackendCase, device: torch.device, tmp_path):
    """A decorated keyword-only call survives compilation, dynamic batch execution, and checkpoint reload."""
    backend = case.factory()
    reference = KeywordOnlyModel(make_mlp(device))
    tuned = ait.Module(KeywordOnlyModel(make_mlp(device)), "keyword-only", strategy=pinned_strategy(backend))

    ait.tune(
        partial(keyword_call, tuned),
        list(make_input(device, batch_size=8).unbind(0)),
        batch_sizes=[1, 4],
        device=device,
        ignore_failing_modules=False,
    )
    selected_backend(tuned, type(backend))
    assert_matches_eager(
        partial(keyword_call, tuned), partial(keyword_call, reference), device, case.tolerance, batch_sizes=(1, 3, 4)
    )

    checkpoint = tmp_path / "keyword-only.ait"
    ait.save(tuned, checkpoint)
    tuned.deactivate()
    restored = ait.load(KeywordOnlyModel(make_mlp(device, seed=1)), checkpoint)
    assert isinstance(restored, ait.Module)
    selected_backend(restored, type(backend))
    assert_matches_eager(
        partial(keyword_call, restored), partial(keyword_call, reference), device, case.tolerance, batch_sizes=(1, 3, 4)
    )
