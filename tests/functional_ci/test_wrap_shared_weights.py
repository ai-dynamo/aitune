# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``ait.wrap`` keeps weights used outside a tuned module on the device after offloading.

The modules come from ``ait.inspect(..., call_tree=True)``, so ``wrap`` finds the call tree through them.

Each toy has the sharing pattern behind a real tuning failure (``tests/toy_models/call_tree_models.py``):

- ``VlaToy``, like Xiaomi-Robotics-0: ``lm_head`` is tied to the LLM's embedding, which is looked up outside both;
- ``GlueToy``, like pi0.5: the root looks up the tuned prefix's embedding outside it; ``DelegatingGlueToy``
  does the same with a tuned module whose ``forward`` runs only one child;
- ``SequentialSubclassToy``, like E5 with sentence-transformers: ``encode`` calls ``.to()`` over the whole model.
- ``DeviceLookupToy``, like Cosmos3-Nano: the pipeline looks up a weight's device outside the tuned module.

Backends that build ahead of time offload the original module after tuning. Without the call tree the toy fails at
the next inference, as the real models do; with it, the shared weights stay and the output matches eager.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import pytest
import torch
from torch import nn

import aitune.torch as ait
from tests.functional_ci.backends import BackendCase, with_backends
from tests.functional_ci.lifecycle import pinned_strategy
from tests.functional_ci.models import seeded
from tests.toy_models.call_tree_models import (
    DelegatingGlueToy,
    DeviceLookupToy,
    DirectReadToy,
    GlueToy,
    SequentialSubclassToy,
    VlaToy,
    vla_inputs,
)


@dataclass
class _Toy:
    model: nn.Module
    call: Callable[..., Any]
    args: tuple
    tuned: Callable[[Any], Any]


def _vla(device: torch.device) -> _Toy:
    model = VlaToy().to(device).eval()
    tokens, image = vla_inputs()
    return _Toy(model, lambda *a: model(*a), (tokens.to(device), image.to(device)), lambda m: m.lm_head)


def _glue(device: torch.device) -> _Toy:
    model = GlueToy().to(device).eval()
    return _Toy(model, lambda *a: model(*a), (torch.randint(0, 16, (2,), device=device),), lambda m: m.prefix)


def _delegating(device: torch.device) -> _Toy:
    model = DelegatingGlueToy().to(device).eval()
    return _Toy(model, lambda *a: model(*a), (torch.randint(0, 16, (2,), device=device),), lambda m: m.holder)


def _encoder(device: torch.device) -> _Toy:
    model = SequentialSubclassToy().to(device).eval()
    return _Toy(model, lambda *a: model.encode(*a), (torch.randn(2, 8, device=device),), lambda m: m[0])


def _direct_read(device):
    model = DirectReadToy().to(device).eval()
    return _Toy(model, lambda *a: model(*a), (torch.randint(0, 16, (2,), device=device),), lambda m: m.unit)


def _device_lookup(device):
    model = DeviceLookupToy().to(device).eval()
    return _Toy(model, lambda *a: model(*a), (torch.randint(0, 16, (2,), device=device),), lambda m: m.unit)


TOYS = {
    "tied-head": _vla,
    "direct-read": _direct_read,
    "device-lookup": _device_lookup,
    "embedding-outside": _glue,
    "embedding-outside-delegating": _delegating,
    "to-from-outside": _encoder,
}


def _wrap_tune_and_run(toy: _Toy, case: BackendCase, device: torch.device, use_call_tree: bool):
    with torch.no_grad():
        info = ait.inspect(
            toy.model,
            [toy.args],
            inference_function=toy.call,
            number_of_iterations=2,
            warmup_iterations=1,
            min_depth=1,
            max_depth=8,
            call_tree=use_call_tree,
        )
    target = toy.tuned(toy.model)
    selected = [m for m in info.get_modules() if m.module is target]
    assert selected, [m.object_path for m in info.get_modules()]
    ait.wrap(toy.model, selected, strategy=pinned_strategy(case.factory()))
    wrapped = cast(ait.Module, toy.tuned(toy.model))
    with torch.no_grad():
        toy.call(*toy.args)  # record the tuned module's inputs
        wrapped.tune(device=device)
        return toy.call(*toy.args)


@pytest.mark.matrix("aot_tuning.models.shared-weights-{toy_id}")
@pytest.mark.parametrize("toy_id", list(TOYS))
@with_backends("export")
def test_wrap_with_call_tree_keeps_shared_weights(toy_id: str, case: BackendCase, device: torch.device):
    """With the call tree, the tuned model runs after offloading and matches eager."""
    with seeded(device, 0):
        toy = TOYS[toy_id](device)
    with torch.no_grad():
        expected = toy.call(*toy.args)

    actual = _wrap_tune_and_run(toy, case, device, use_call_tree=True)

    case.tolerance.assert_close(actual, expected)


@pytest.mark.matrix("aot_tuning.models.shared-weights-{toy_id}")
@pytest.mark.parametrize("toy_id", list(TOYS))
@with_backends("export")
def test_wrap_without_call_tree_breaks_on_shared_weights(toy_id: str, case: BackendCase, device: torch.device):
    """Without the call tree, offloading takes the shared weights away, as on the real models."""
    with seeded(device, 0):
        toy = TOYS[toy_id](device)

    with pytest.raises(Exception) as error:
        _wrap_tune_and_run(toy, case, device, use_call_tree=False)
    # PyTorch raises RuntimeError or NotImplementedError; ONNX Runtime raises a plain Exception for a meta input.
    assert isinstance(error.value, (RuntimeError, NotImplementedError)) or "Unsupported device type" in str(error.value)
