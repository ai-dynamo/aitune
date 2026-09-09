# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
import torch

from aitune.torch.task.correctness import CorrectnessInputMutationError
from aitune.torch.task.stateful_validation import StatefulValidation, _preserve_buffers
from tests.utilities.helpers import make_recorded_graph_spec


class Accumulator(torch.nn.Module):
    def forward(self, value, cache):
        cache["value"].add_(value)
        cache["counter"].add_(1)
        return {"value": cache["value"], "cache": cache}


@pytest.fixture
def validation(tmp_path):
    model = Accumulator()
    sample = ((torch.ones(2), {"value": torch.zeros(2), "counter": torch.tensor(0)}), {})
    spec = make_recorded_graph_spec(model, sample, strict=True)
    stateful_validation = StatefulValidation(tmp_path, spec)
    stateful_validation.record(model, [sample])
    return stateful_validation, sample


def test_stateful_validation_retains_state_and_preserves_samples(validation):
    stateful_validation, sample = validation
    counters = []

    def infer(value, cache):
        counters.append(cache["counter"].item())
        return Accumulator()(value, cache)

    stateful_validation.check(infer, torch.device("cpu"), name="candidate")
    stateful_validation.check(infer, torch.device("cpu"), name="reloaded")
    assert counters == [0, 1, 2, 0, 1, 2]
    assert sample[0][1]["counter"].item() == 0
    torch.testing.assert_close(sample[0][1]["value"], torch.zeros(2))


@pytest.mark.parametrize("defect", ["values", "second-call", "alias", "replacement", "rank", "dtype"])
def test_stateful_validation_rejects_contract_defects(validation, defect):
    stateful_validation, _ = validation

    def infer(value, cache):
        output = Accumulator()(value, cache)
        if defect == "values":
            cache["value"].zero_()
        elif defect == "second-call" and cache["counter"].item() > 1:
            cache["value"].zero_()
        elif defect == "alias":
            output["cache"] = deepcopy(cache)
        elif defect == "replacement":
            cache["counter"] = cache["counter"].clone()
        elif defect == "rank":
            cache["counter"] = cache["counter"].reshape(1)
        elif defect == "dtype":
            cache["counter"] = cache["counter"].float()
        return output

    with pytest.raises(CorrectnessInputMutationError, match="differs? from eager"):
        stateful_validation.check(infer, torch.device("cpu"), name="broken")


def test_stateful_validation_checks_every_recorded_sample(tmp_path):
    model = Accumulator()
    samples = [
        ((torch.full((2,), value), {"value": torch.zeros(2), "counter": torch.tensor(0)}), {}) for value in (1.0, 7.0)
    ]
    stateful_validation = StatefulValidation(tmp_path, make_recorded_graph_spec(model, samples[0]))
    stateful_validation.record(model, samples)

    def wrong_for_second(value, cache):
        return model(torch.ones_like(value), cache)

    with pytest.raises(CorrectnessInputMutationError, match="sample 1"):
        stateful_validation.check(wrong_for_second, torch.device("cpu"), name="candidate")


def test_stateful_validation_preserves_cross_argument_aliases(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, cache, *, same_cache):
            assert cache is same_cache
            cache[0].add_(1)
            return cache

    cache = [torch.zeros(2)]
    sample = ((cache,), {"same_cache": cache})
    model = Model()
    stateful_validation = StatefulValidation(tmp_path, make_recorded_graph_spec(model, sample))
    stateful_validation.record(model, [sample])
    stateful_validation.check(model, torch.device("cpu"), name="candidate")
    torch.testing.assert_close(cache[0], torch.zeros(2))


def test_reference_buffers_are_restored_even_on_failure():
    model = torch.nn.Module()
    original = torch.zeros(2)
    model.register_buffer("state", original)
    with pytest.raises(RuntimeError, match="failure"), _preserve_buffers(model):
        model.state.add_(1)
        model.state = torch.ones(3)
        model.register_buffer("temporary", torch.ones(1))
        raise RuntimeError("failure")
    assert model.state is original
    assert "temporary" not in model._buffers
    torch.testing.assert_close(original, torch.zeros(2))


def test_stateful_validation_requires_consecutive_calls(tmp_path):
    with pytest.raises(ValueError, match="at least two"):
        StatefulValidation(tmp_path, None, steps=1)
