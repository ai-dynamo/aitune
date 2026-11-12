# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Caller-visible input state changes through tuning and checkpoint reload."""

from functools import partial
from typing import cast

import pytest
import torch
from torch import nn

import aitune.torch as ait
from aitune.torch.backend import TensorRTBackend
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.patched_module import ModuleState, PatchedModule
from aitune.torch.jit.patcher import prepare_for_jit_tuning
from aitune.torch.module.tuned_module import TunedModule
from tests.functional_ci.backends import BackendCase, tensorrt_backend, with_backends
from tests.functional_ci.lifecycle import TENSORRT_FP32, pinned_strategy, selected_backend
from tests.functional_ci.models import seeded
from tests.toy_models.stateful import update_tensor

FEATURES = 4


class InplaceCacheModel(nn.Module):
    """Accumulate into a caller-provided cache without returning that cache."""

    def __init__(self, device: torch.device, seed: int = 0):
        super().__init__()
        self.register_buffer("increment", torch.tensor(float(seed + 1), device=device))

    def forward(self, x: torch.Tensor, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state["cache"].add_(self.increment)
        return {"value": x + state["cache"]}


class ReplacingCacheModel(nn.Module):
    """Replace the cached tensor and return the caller's state container."""

    def __init__(self, device: torch.device, seed: int = 0):
        super().__init__()
        self.register_buffer("increment", torch.tensor(float(seed + 1), device=device))

    def forward(self, x: torch.Tensor, state: dict[str, torch.Tensor]):
        state["cache"] = state["cache"] + self.increment
        return {"value": x + state["cache"], "state": state}


class TensorMutationModel(nn.Module):
    """Exercise indexed writes and arithmetic updates without returning the cache."""

    def __init__(self, device: torch.device, seed: int = 0, *, operation: str):
        super().__init__()
        self.operation = operation
        self.register_buffer("scale", torch.tensor(float(seed + 1), device=device))

    def forward(self, x: torch.Tensor, state: dict[str, torch.Tensor]):
        index = torch.arange(x.shape[0], device=x.device).flip(0)
        if self.operation == "scatter_":
            index = index[:, None].expand_as(x)
        update_tensor(state["cache"], x * self.scale, index, self.operation)
        return {"value": x + state["cache"]}


class ProjectedCacheModel(nn.Module):
    """A weighted block whose caller supplies the accumulated state."""

    def __init__(self, device: torch.device, seed: int = 0, *, replace_cache: bool = False):
        super().__init__()
        with seeded(device, seed):
            self.projection = nn.Linear(FEATURES, FEATURES, bias=False).to(device)
        self.replace_cache = replace_cache

    def forward(self, x, state):
        projected = self.projection(x)
        if self.replace_cache:
            state["cache"] = state["cache"] + projected
        else:
            state["cache"].add_(projected)
        return {"value": projected + state["cache"]}


class ImmutableContainerCacheModel(nn.Module):
    """Copy updates into a direct tensor argument or a tensor below an immutable tuple."""

    def __init__(self, device: torch.device, seed: int = 0):
        super().__init__()
        self.register_buffer("increment", torch.tensor(float(seed + 1), device=device))

    def forward(self, x, state):
        tensor = state[0] if isinstance(state, tuple) else state
        tensor.add_(x * self.increment)
        return x + tensor


class LazyCacheModel(nn.Module):
    """Initialize a caller's empty cache, then accumulate tensor state during decode."""

    def __init__(self, device: torch.device, seed: int = 0):
        super().__init__()
        with seeded(device, seed):
            self.projection = nn.Linear(FEATURES, FEATURES).to(device)

    def forward(self, x, state):
        projected = self.projection(x)
        state[0] = projected if state[0] is None else state[0] + projected
        return {"value": projected + state[0], "state": state}


def check_mutable_input_lifecycle(
    case: BackendCase, device: torch.device, tmp_path, model_factory, returns_state, batch_sizes=(2,)
):
    """Outputs, accumulated input state, and container aliases survive tuning and save/load."""
    backend = case.factory()
    tuned = ait.Module(model_factory(device), "mutable-inputs", strategy=pinned_strategy(backend))

    ait.tune(
        partial(mutable_input_pipeline, tuned),
        list(torch.full((8, FEATURES), 2.0, device=device).unbind(0)),
        batch_sizes=list(batch_sizes),
        device=device,
        ignore_failing_modules=False,
    )

    selected_backend(tuned, type(backend))
    for batch_size in batch_sizes:
        check_mutable_input_state(tuned, case, device, model_factory, returns_state, batch_size)
    checkpoint = tmp_path / "mutable_inputs.ait"
    ait.save(tuned, checkpoint)
    tuned.deactivate()
    # A different initial increment ensures reload restores the model buffer as well as mutation metadata.
    restored = ait.load(model_factory(device, seed=1), checkpoint)
    assert isinstance(restored, ait.Module)
    selected_backend(restored, type(backend))
    for batch_size in batch_sizes:
        check_mutable_input_state(restored, case, device, model_factory, returns_state, batch_size)
    return restored


def check_jit_mutations(case, device, *, replace_cache=False, model_factory=None, batch_sizes=(1, 2)):
    backend = case.factory()
    jit_config.mode = JITMode.TUNE_EAGER
    jit_config.device = device
    jit_config.min_samples = 2
    jit_config.batch_axis_required = len(set(batch_sizes)) > 1
    jit_config.strategy = pinned_strategy(backend)
    if model_factory is None:
        model_factory = partial(ProjectedCacheModel, replace_cache=replace_cache)
    with prepare_for_jit_tuning():
        model = model_factory(device).eval()

    with torch.no_grad():
        for batch in (batch_sizes[0], batch_sizes[-1]):
            value = torch.ones(batch, FEATURES, device=device)
            model(value, {"cache": torch.full_like(value, 0.25)})

        (head,) = PatchedModule.heads
        assert head._state == ModuleState.TUNED
        (selected,) = cast(TunedModule, head._wrapper).backends.values()
        assert type(selected) is type(backend)
        for batch_size in batch_sizes:
            check_mutable_input_state(model, case, device, model_factory, returns_state=False, batch_size=batch_size)


def mutable_input_pipeline(model, value: torch.Tensor):
    # Nonzero state and inputs expose every operation, including multiplication and zeroing.
    # Each recording call starts fresh; inference below must retain updates between calls.
    state = {"cache": torch.full_like(value, 0.25)}
    return model(value, state)


def check_mutable_input_state(model, case, device, model_factory, returns_state, batch_size):
    eager = model_factory(device).eval()
    # Interleave independent callers. A later inference must not overwrite another cache.
    states = [{"cache": torch.full((batch_size, FEATURES), initial, device=device)} for initial in (0.25, 1.5)]
    eager_states = [{"cache": state["cache"].clone()} for state in states]
    retained_outputs = []
    with torch.no_grad():
        for cache_index, scale in ((0, 1.0), (1, 7.0), (0, 3.0)):
            state = states[cache_index]
            value = torch.arange(batch_size * FEATURES, device=device).reshape(batch_size, FEATURES).float() / 8 + scale
            expected = eager(value, eager_states[cache_index])
            actual = model(value, state)
            retained_outputs.append((actual["value"], expected["value"].clone()))
            for retained, expected_value in retained_outputs:
                case.tolerance.assert_close(retained, expected_value)
            for actual_state, expected_state in zip(states, eager_states, strict=True):
                case.tolerance.assert_close(actual_state["cache"], expected_state["cache"])
            # Backends may install their own result tensors; container identity is the shared contract.
            if returns_state:
                assert actual["state"] is state


def wrap_mutable_tensor(tensor, container):
    return (tensor,) if container == "tuple" else tensor


def immutable_container_pipeline(model, container, value):
    return model(value, wrap_mutable_tensor(torch.full_like(value, 0.25), container))


def check_immutable_container_mutations(model, case, device, container, backend_type):
    selected_backend(model, backend_type)
    reference = ImmutableContainerCacheModel(device)
    original = torch.full((2, FEATURES), 0.25, device=device)
    state = wrap_mutable_tensor(original, container)
    expected_state = wrap_mutable_tensor(original.clone(), container)
    with torch.no_grad():
        for scale in (1.0, 3.0):
            value = torch.full_like(original, scale)
            expected = reference(value, expected_state)
            case.tolerance.assert_close(model(value, state), expected)
            assert (state[0] if container == "tuple" else state) is original
            expected_tensor = expected_state[0] if container == "tuple" else expected_state
            case.tolerance.assert_close(original, expected_tensor)


def lazy_cache_pipeline(model, value):
    state = [None]
    model(value, state)
    return model(value + 1, state)


def check_lazy_mutations(model, case, device, backend_type):
    backends = cast(TunedModule, model.module).backends.values()
    assert [type(selected) for selected in backends] == [backend_type] * 2
    reference = LazyCacheModel(device).eval()
    states, expected_states = [[None], [None]], [[None], [None]]
    with torch.no_grad():
        for index, scale in ((0, 1.0), (1, 7.0), (0, 3.0), (1, 5.0)):
            value = torch.full((2, FEATURES), scale, device=device)
            expected = reference(value, expected_states[index])
            actual = model(value, states[index])
            assert actual["state"] is states[index]
            case.tolerance.assert_close(actual["value"], expected["value"])
            for state, expected_state in zip(states, expected_states, strict=True):
                case.tolerance.assert_close(state[0], expected_state[0])


@pytest.mark.matrix("aot_tuning.models.mutable-inputs")
@pytest.mark.parametrize("model_factory", [InplaceCacheModel, ProjectedCacheModel], ids=["buffer", "projected"])
@pytest.mark.parametrize("batch_sizes", [(2,), (1, 2)], ids=["static", "dynamic"])
@with_backends(
    "base",
    marks={
        "torch-tensorrt-jit": pytest.mark.xfail(
            reason="Torch-TensorRT JIT drops in-place input writes; related upstream fix: pytorch/TensorRT#4894",
            raises=AssertionError,
            strict=True,
        ),
    },
)
def test_inplace_mutable_input_lifecycle(case: BackendCase, device: torch.device, tmp_path, model_factory, batch_sizes):
    """Unreturned in-place cache updates survive consecutive calls and save/load."""
    check_mutable_input_lifecycle(case, device, tmp_path, model_factory, returns_state=False, batch_sizes=batch_sizes)


@pytest.mark.matrix("aot_tuning.models.mutable-inputs")
@with_backends("base")
def test_replacing_mutable_input_lifecycle(case: BackendCase, device: torch.device, tmp_path):
    """Cache replacement and the returned container alias survive consecutive calls and save/load."""
    check_mutable_input_lifecycle(case, device, tmp_path, ReplacingCacheModel, returns_state=True)


@pytest.mark.matrix("aot_tuning.models.mutable-inputs")
@pytest.mark.parametrize("operation", ["mul_", "index_copy_", "scatter_", "zero_"])
@with_backends(
    "base",
    marks={
        "torch-tensorrt-jit": pytest.mark.skip(
            reason="Torch-TensorRT JIT drops in-place input writes; related upstream fix: pytorch/TensorRT#4894",
        ),
    },
)
def test_tensor_mutation_lifecycle(case: BackendCase, device: torch.device, tmp_path, operation):
    """Numerical input updates survive repeated calls, interleaved caches, and checkpoint reload."""
    check_mutable_input_lifecycle(
        case, device, tmp_path, partial(TensorMutationModel, operation=operation), returns_state=False
    )


@pytest.mark.matrix("jit_tuning.models.mutable-inputs")
@pytest.mark.parametrize(
    "model_factory",
    [
        pytest.param(
            InplaceCacheModel,
            id="buffer",
            marks=pytest.mark.skip(reason="Automatic JIT tuning excludes models without parameters"),
        ),
        pytest.param(ProjectedCacheModel, id="projected"),
    ],
)
@with_backends(
    "base",
    marks={
        "torch-tensorrt-jit": pytest.mark.skip(
            reason="Torch-TensorRT JIT drops in-place input writes; related upstream fix: pytorch/TensorRT#4894",
        ),
    },
)
def test_jit_dynamic_inplace_mutable_inputs(case: BackendCase, device: torch.device, model_factory):
    """Automatic JIT tuning preserves dynamic-batch updates using the same models and checks as AOT."""
    check_jit_mutations(case, device, model_factory=model_factory, batch_sizes=(1, 2))


@pytest.mark.matrix("jit_tuning.models.mutable-inputs")
@pytest.mark.parametrize(
    "model_factory",
    [
        pytest.param(
            InplaceCacheModel,
            id="buffer",
            marks=pytest.mark.skip(reason="Automatic JIT tuning excludes models without parameters"),
        ),
        pytest.param(ProjectedCacheModel, id="projected"),
    ],
)
@with_backends(
    "base",
    marks={
        "torch-tensorrt-jit": pytest.mark.skip(
            reason="Torch-TensorRT JIT drops in-place input writes; related upstream fix: pytorch/TensorRT#4894",
        ),
    },
)
def test_jit_static_inplace_mutable_inputs(case: BackendCase, device: torch.device, model_factory):
    """Automatic JIT tuning preserves fixed-batch updates using the same models and checks as AOT."""
    check_jit_mutations(case, device, model_factory=model_factory, batch_sizes=(2,))


@pytest.mark.matrix("jit_tuning.models.mutable-inputs")
@with_backends("base")
def test_jit_replacing_mutable_inputs(case: BackendCase, device: torch.device):
    """Automatic JIT tuning preserves tensor replacement and independent callers' accumulated state."""
    check_jit_mutations(case, device, replace_cache=True)


@pytest.mark.matrix("other.cuda-graphs-mutable-inputs/functional-ci")
def test_cuda_graph_mutable_input_lifecycle(device: torch.device, tmp_path):
    """Captured CUDA graph outputs and state remain valid across interleaved callers and reload."""
    case = BackendCase(
        factory=partial(tensorrt_backend, use_cuda_graphs=True),
        tolerance=TENSORRT_FP32,
        tags=frozenset({"base"}),
    )
    restored = check_mutable_input_lifecycle(case, device, tmp_path, InplaceCacheModel, returns_state=False)
    backend = selected_backend(restored, TensorRTBackend)
    assert isinstance(backend, TensorRTBackend)
    # Requesting CUDA graphs alone does not prove capture: fallback must not satisfy this case.
    assert backend._use_cuda_graphs
    assert backend._cuda_graphs.active is not None
    assert backend._cuda_graphs.active.graph is not None


@pytest.mark.matrix("aot_tuning.models.mutable-inputs")
@pytest.mark.parametrize("container", ["direct", "tuple"])
@with_backends(
    "base",
    marks={
        "torch-tensorrt-jit": pytest.mark.skip(
            reason="Torch-TensorRT JIT drops in-place input writes; related upstream fix: pytorch/TensorRT#4894",
        ),
    },
)
def test_immutable_container_mutations(case: BackendCase, device: torch.device, tmp_path, container):
    """The original caller tensor receives updates through direct and tuple arguments after tuning and reload."""

    backend = case.factory()
    tuned = ait.Module(ImmutableContainerCacheModel(device), "immutable-state", strategy=pinned_strategy(backend))

    ait.tune(
        partial(immutable_container_pipeline, tuned, container),
        list(torch.ones(8, FEATURES, device=device).unbind(0)),
        batch_sizes=[2],
        device=device,
        ignore_failing_modules=False,
    )

    check_immutable_container_mutations(tuned, case, device, container, type(backend))
    checkpoint = tmp_path / "immutable-state.ait"
    ait.save(tuned, checkpoint)
    tuned.deactivate()
    restored = ait.load(ImmutableContainerCacheModel(device, seed=1), checkpoint)
    assert isinstance(restored, ait.Module)
    check_immutable_container_mutations(restored, case, device, container, type(backend))


@pytest.mark.matrix("aot_tuning.models.mutable-inputs", "aot_tuning.models.multiple-graphs")
@with_backends("base")
def test_lazy_mutable_input_lifecycle(case: BackendCase, device: torch.device, tmp_path):
    """Initialization and decode select distinct graphs, restore aliases, and survive checkpoint reload."""
    backend = case.factory()
    tuned = ait.Module(LazyCacheModel(device).eval(), "lazy-cache", strategy=pinned_strategy(backend))

    ait.tune(
        partial(lazy_cache_pipeline, tuned),
        list(torch.ones(8, FEATURES, device=device).unbind(0)),
        batch_sizes=[2],
        device=device,
        ignore_failing_modules=False,
    )

    # Recording metadata is available before saving; reload restores the compiled backends.
    assert len(tuned.graph_specs) == 2
    check_lazy_mutations(tuned, case, device, type(backend))
    checkpoint = tmp_path / "lazy-cache.ait"
    ait.save(tuned, checkpoint)
    tuned.deactivate()
    restored = ait.load(LazyCacheModel(device, seed=1).eval(), checkpoint)
    assert isinstance(restored, ait.Module)
    check_lazy_mutations(restored, case, device, type(backend))
