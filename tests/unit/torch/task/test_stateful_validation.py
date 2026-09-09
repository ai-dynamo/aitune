# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from aitune.torch.backend import (
    ONNXRuntimeBackend,
    TensorRTBackend,
    TorchEagerBackend,
    TorchInductorJitBackend,
    TorchTensorRTAotBackend,
    TorchTensorRTJitBackend,
)
from aitune.torch.module.forward_contract import TensorMutationPolicy
from aitune.torch.task.correctness import CorrectnessInputMutationError
from aitune.torch.task.stateful_validation import StatefulValidation, preserve_buffers
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


@pytest.mark.parametrize("backend_class", [ONNXRuntimeBackend, TensorRTBackend])
def test_stateful_validation_accepts_engine_owned_state(validation, backend_class):
    stateful_validation, _ = validation
    backend = backend_class()
    spec = backend._graph_spec = stateful_validation.graph_spec

    def infer(value, cache):
        engine_cache = {"value": cache["value"] + value, "counter": cache["counter"] + 1}
        output = {"value": engine_cache["value"], "cache": engine_cache}
        tensors = spec.output_spec.flatten_tensors(output)
        outputs = dict(zip(spec.output_names(), tensors, strict=True))
        arguments = {"value": value, "cache": cache}
        if isinstance(backend, ONNXRuntimeBackend):
            result = backend._prepare_outputs(outputs, arguments)
        else:
            backend._output_allocator = SimpleNamespace(outputs=outputs)
            result = backend._prepare_outputs_for_return(arguments)
        assert cache["value"] is engine_cache["value"]
        assert result["cache"] is cache
        return result

    with pytest.raises(CorrectnessInputMutationError, match="aliases differ"):
        stateful_validation.check(infer, torch.device("cpu"), name="strict")
    stateful_validation.check(
        infer, torch.device("cpu"), name="engine", tensor_mutation_policy=backend.tensor_mutation_policy
    )


@pytest.mark.parametrize("defect", ["values", "second-call", "alias", "rank", "dtype"])
def test_replacement_policy_still_rejects_contract_defects(validation, defect):
    stateful_validation, _ = validation

    def infer(value, cache):
        cache["value"] = cache["value"] + value
        cache["counter"] = cache["counter"] + 1
        output = {"value": cache["value"], "cache": cache}
        if defect == "values" or (defect == "second-call" and cache["counter"].item() > 1):
            cache["value"].zero_()
        elif defect == "alias":
            output["cache"] = deepcopy(cache)
        elif defect == "rank":
            cache["counter"] = cache["counter"].reshape(1)
        elif defect == "dtype":
            cache["counter"] = cache["counter"].float()
        return output

    with pytest.raises(CorrectnessInputMutationError, match="differs? from eager"):
        stateful_validation.check(
            infer, torch.device("cpu"), name="broken", tensor_mutation_policy=TensorMutationPolicy.REPLACE_INPUT
        )


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
    with pytest.raises(RuntimeError, match="failure"), preserve_buffers(model):
        model.state.add_(1)
        model.state = torch.ones(3)
        model.register_buffer("temporary", torch.ones(1))
        raise RuntimeError("failure")
    assert model.state is original
    assert "temporary" not in model._buffers
    torch.testing.assert_close(original, torch.zeros(2))


def test_stateful_validation_requires_consecutive_calls(tmp_path, validation):
    stateful_validation, _ = validation
    with pytest.raises(ValueError, match="at least two"):
        StatefulValidation(tmp_path, stateful_validation.graph_spec, steps=1)


def test_cuda_graph_state_survives_interleaved_caches(tmp_path):
    model = Accumulator()
    sample = ((torch.ones(2), {"value": torch.zeros(2), "counter": torch.tensor(0)}), {})
    spec = make_recorded_graph_spec(model, sample, strict=True)
    backend = TensorRTBackend()
    backend._config.use_cuda_graphs = True
    backend._graph_spec = spec
    buffers = (torch.empty(2), torch.empty(2), torch.tensor(0))

    def infer(value, cache):
        updated = cache["value"] + value
        counter = cache["counter"] + 1
        for target, source in zip(buffers, (updated, updated, counter), strict=True):
            target.copy_(source)
        backend._output_allocator = SimpleNamespace(outputs=dict(zip(spec.output_names(), buffers, strict=True)))
        return backend._prepare_outputs_for_return({"value": value, "cache": cache})

    caches = [deepcopy(sample[0][1]), deepcopy(sample[0][1])]
    for index, scale in ((0, 1), (1, 7), (0, 2)):
        result = infer(torch.full((2,), float(scale)), caches[index])
        assert result["cache"] is caches[index]
    torch.testing.assert_close(caches[0]["value"], torch.full((2,), 3.0))
    torch.testing.assert_close(caches[1]["value"], torch.full((2,), 7.0))
    assert caches[0]["counter"].item() == 2
    assert caches[1]["counter"].item() == 1


@pytest.mark.parametrize("backend_class", [ONNXRuntimeBackend, TensorRTBackend])
@pytest.mark.parametrize("container", [lambda tensor: tensor, lambda tensor: (tensor,)])
def test_engine_mutations_update_direct_and_tuple_inputs(tmp_path, backend_class, container):
    class Model(torch.nn.Module):
        def forward(self, value, state):
            tensor = state[0] if isinstance(state, tuple) else state
            tensor.add_(value)
            return tensor

    model = Model()
    sample = ((torch.ones(2), container(torch.zeros(2))), {})
    spec = make_recorded_graph_spec(model, sample, strict=True)
    backend = backend_class()
    backend._graph_spec = spec

    def infer(value, state):
        tensor = state[0] if isinstance(state, tuple) else state
        outputs = dict(zip(spec.output_names(), (tensor + value,), strict=True))
        arguments = {"value": value, "state": state}
        if isinstance(backend, ONNXRuntimeBackend):
            return backend._prepare_outputs(outputs, arguments)
        backend._output_allocator = SimpleNamespace(outputs=outputs)
        return backend._prepare_outputs_for_return(arguments)

    validation = StatefulValidation(tmp_path, spec)
    validation.record(model, [sample])
    validation.check(
        infer, torch.device("cpu"), name="direct-input", tensor_mutation_policy=backend.tensor_mutation_policy
    )


class BufferedAccumulator(Accumulator):
    def __init__(self):
        super().__init__()
        self.register_buffer("calls", torch.tensor(0))

    def forward(self, value, cache):
        self.calls.add_(1)
        output = super().forward(value, cache)
        return {"state": output, "calls": self.calls.clone()}


def test_validation_preserves_buffers_across_multiple_samples_and_checks(tmp_path):
    model = BufferedAccumulator()
    sample = ((torch.ones(2), {"value": torch.zeros(2), "counter": torch.tensor(0)}), {})
    spec = make_recorded_graph_spec(model, sample)
    initial = model.calls.clone()
    validation = StatefulValidation(tmp_path, spec)
    validation.record(model, [sample, sample])
    torch.testing.assert_close(model.calls, initial)
    for name in ("compiled", "reloaded"):
        validation.check(model, torch.device("cpu"), name=name)
        torch.testing.assert_close(model.calls, initial)


@pytest.mark.parametrize(
    "backend_class", [TorchEagerBackend, TorchInductorJitBackend, TorchTensorRTJitBackend, TorchTensorRTAotBackend]
)
def test_backend_validation_restores_buffers_on_failure(mocker, backend_class):
    model = BufferedAccumulator()
    if backend_class in (TorchTensorRTJitBackend, TorchTensorRTAotBackend):
        mocker.patch(f"{backend_class.__module__}.assert_cuda_is_available")
        mocker.patch(f"{backend_class.__module__}.assert_torch_tensorrt")
    backend = backend_class()
    if isinstance(backend, TorchTensorRTAotBackend):
        backend._opt_module = model
    else:
        backend._orig_module = model
    with pytest.raises(RuntimeError, match="validation failed"), backend.validation_context():
        model.calls.add_(3)
        raise RuntimeError("validation failed")
    assert model.calls.item() == 0
