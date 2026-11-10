# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Call signatures, structured inputs, and structured outputs across backends."""

from collections.abc import Callable
from dataclasses import replace
from functools import wraps

import pytest
import torch
from torch import nn

import aitune.torch as ait
from tests.functional_ci.backends import BackendCase, with_backends
from tests.functional_ci.lifecycle import assert_matches_eager, pinned_strategy, selected_backend
from tests.functional_ci.models import IN_FEATURES, make_input, make_mlp

CALL_LAYOUTS = {
    "positional": lambda model, x, residual: model(x, residual),
    "mixed": lambda model, x, residual: model(x, residual=residual),
    "keyword": lambda model, x, residual: model(x=x, residual=residual),
    "reordered-keyword": lambda model, x, residual: model(residual=residual, x=x),
}


def _require_keywords(forward):
    """Keep the original signature while requiring keyword arguments at runtime."""

    @wraps(forward)
    def wrapped(self, *args, **kwargs):
        if args:
            raise TypeError("This forward decorator requires keyword arguments")
        return forward(self, **kwargs)

    return wrapped


class KeywordCallModel(nn.Module):
    """An MLP whose decorated forward requires calls to retain their keyword layout."""

    def __init__(self, device: torch.device, seed: int = 0):
        super().__init__()
        self.mlp = make_mlp(device, seed)

    @_require_keywords
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SignatureModel(nn.Module):
    """An MLP with two required arguments that accept positional or keyword calls."""

    def __init__(self, device: torch.device, seed: int = 0):
        super().__init__()
        self.mlp = make_mlp(device, seed)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # Distinct roles make argument swaps observable in the output.
        return self.mlp(x + 0.5 * residual)


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
        return {"out": hidden, "stats": (hidden.sum(-1), hidden.amax(-1))}


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


@pytest.mark.matrix(
    "aot_tuning.models.non-tensor-inputs",
    "aot_tuning.models.nested-inputs",
    "aot_tuning.models.structured-outputs",
)
@with_backends("base")
def test_structured_io(case: BackendCase, device: torch.device):
    """Nested dict and list inputs, a float and a `None` argument, and a nested dict/tuple output survive tuning."""
    backend = case.factory()
    # Row sums in `stats` accumulate the per-element error over `OUT_FEATURES` values, so widen `atol`.
    tolerance = replace(case.tolerance, atol=5 * case.tolerance.atol)
    bias = torch.linspace(-1, 1, IN_FEATURES, device=device)
    residuals = [bias.flip(0), 0.5 * bias]
    reference = StructuredModel(make_mlp(device))
    tuned = ait.Module(StructuredModel(make_mlp(device)), "structured", strategy=pinned_strategy(backend))

    def pipeline(model: Callable) -> Callable:
        # Every tensor in the sample is batched, so the bias and residuals are repeated per row.
        return lambda x: model(
            x,
            {"bias": bias.repeat(x.shape[0], 1)},
            [residual.repeat(x.shape[0], 1) for residual in residuals],
            scale=2.0,
            mask=None,
        )

    ait.tune(pipeline(tuned), list(make_input(device, batch_size=8).unbind(0)), batch_sizes=[1, 4], device=device)

    selected_backend(tuned, type(backend))
    assert_matches_eager(pipeline(tuned), pipeline(reference), device, tolerance, batch_sizes=(1, 3, 4))


@pytest.mark.matrix("aot_tuning.models.call-layouts")
@with_backends("base")
@pytest.mark.parametrize("record_layout", ["positional", "mixed", "keyword", "reordered-keyword"])
def test_equivalent_call_layouts(case: BackendCase, device: torch.device, tmp_path, record_layout: str):
    """Equivalent calls match eager before and after reload, regardless of the recorded call layout."""
    backend = case.factory()
    reference = SignatureModel(device)
    tuned = ait.Module(SignatureModel(device), "call-layouts", strategy=pinned_strategy(backend))
    record_call = CALL_LAYOUTS[record_layout]

    ait.tune(
        lambda value: record_call(tuned, value, value.flip(-1)),
        list(make_input(device, batch_size=8).unbind(0)),
        batch_sizes=[1, 4],
        device=device,
        ignore_failing_modules=False,
    )

    def check_layouts(model):
        selected_backend(model, type(backend))
        # Use one compiled graph for every layout; unrecorded batch 3 also exercises dynamic inference.
        for batch_size in (1, 3, 4):
            value = make_input(device, batch_size=batch_size, seed=5)
            residual = value.flip(-1)
            with torch.no_grad():
                expected = reference(value, residual)
                for layout, call in CALL_LAYOUTS.items():
                    try:
                        actual = call(model, value, residual)
                    except Exception as error:
                        raise AssertionError(
                            f"Inference layout {layout!r} failed after recording {record_layout!r}"
                        ) from error
                    case.tolerance.assert_close(actual, expected)

    check_layouts(tuned)
    checkpoint = tmp_path / "call_layouts.ait"
    ait.save(tuned, checkpoint)
    tuned.deactivate()
    # Different initial weights verify that reload restores weights as well as the compiled call contract.
    restored = ait.load(SignatureModel(device, seed=1), checkpoint)
    assert isinstance(restored, ait.Module)
    check_layouts(restored)


@pytest.mark.matrix("aot_tuning.models.keyword-calls")
@with_backends("base")
def test_keyword_call_layout(case: BackendCase, device: torch.device, tmp_path):
    """Keyword calls survive recording, boundary validation, backend inference, and checkpoint reload."""
    backend = case.factory()
    reference = KeywordCallModel(device)
    tuned = ait.Module(KeywordCallModel(device), "keyword-calls", strategy=pinned_strategy(backend))

    ait.tune(
        lambda value: tuned(x=value),
        list(make_input(device, batch_size=8).unbind(0)),
        batch_sizes=[1, 4],
        device=device,
        ignore_failing_modules=False,
    )

    selected_backend(tuned, type(backend))
    assert_matches_eager(
        lambda value: tuned(x=value),
        lambda value: reference(x=value),
        device,
        case.tolerance,
        batch_sizes=(1, 3, 4),
    )

    checkpoint = tmp_path / "keyword_calls.ait"
    ait.save(tuned, checkpoint)
    tuned.deactivate()
    # Different initial weights ensure that loading also restores the saved model weights.
    restored = ait.load(KeywordCallModel(device, seed=1), checkpoint)
    assert isinstance(restored, ait.Module)
    selected_backend(restored, type(backend))
    assert_matches_eager(
        lambda value: restored(x=value),
        lambda value: reference(x=value),
        device,
        case.tolerance,
        batch_sizes=(1, 3, 4),
    )


@pytest.mark.matrix("aot_tuning.models.nested-inputs")
@with_backends("base")
@pytest.mark.parametrize("batch_sizes", [[2], [2, 4]], ids=["static", "dynamic"])
def test_nested_tensor_inputs(case: BackendCase, batch_sizes: list[int], device: torch.device):
    """Tune nested tensor inputs passed as kwargs at static and dynamic batch, as in legacy 044.

    The inputs are a dict, a list, an unbatched tensor, and a 1-D batched scalar. `test_structured_io` covers only a
    nested dict and list, together with non-tensor arguments.
    """
    backend = case.factory()
    bias = torch.linspace(-1, 1, IN_FEATURES, device=device)
    residuals = [bias.flip(0), 0.5 * bias]
    reference = NestedInputModel(make_mlp(device))
    tuned = ait.Module(NestedInputModel(make_mlp(device)), "nested", strategy=pinned_strategy(backend))

    def pipeline(model: Callable) -> Callable:
        # `positions` keeps one shape at every batch size, like `img_ids` in 044; the other tensors are batched.
        return lambda x: model(
            x,
            extras={"bias": bias.repeat(x.shape[0], 1)},
            residuals=[residual.repeat(x.shape[0], 1) for residual in residuals],
            positions=bias.flip(0),
            timestep=torch.linspace(0.5, 1.5, x.shape[0], device=device),
        )

    ait.tune(pipeline(tuned), list(make_input(device, batch_size=8).unbind(0)), batch_sizes=batch_sizes, device=device)

    selected_backend(tuned, type(backend))
    check_sizes = range(batch_sizes[0], batch_sizes[-1] + 1)
    assert_matches_eager(pipeline(tuned), pipeline(reference), device, case.tolerance, batch_sizes=check_sizes)
