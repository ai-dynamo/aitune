# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for wrapper module."""

import functools
import inspect
from collections import OrderedDict
from copy import deepcopy
from unittest.mock import Mock, call

import pytest
import torch

from aitune.torch import TuneStrategy
from aitune.torch.backend.torch_eager import TorchEagerBackend
from aitune.torch.backend.torch_inductor_jit_backend import TorchInductorJitBackend
from aitune.torch.config import aitune_cache_dir
from aitune.torch.config import config as global_config
from aitune.torch.distributed import resolve_tuning_device
from aitune.torch.dynamic_shapes import BatchDim
from aitune.torch.module.forward_signature import ForwardSignature
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.module.sample_store import SampleStore
from aitune.torch.module.tuned_module import TunedModule
from aitune.torch.module.wrapper_module import Module, ModuleState
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.task.profiling import (
    AllSamplesProfilingStopStrategy,
    ModelExecutionTimeMeasuringStrategy,
    NumStepsMeasuringStopStrategy,
    ProfilingConfig,
)
from aitune.torch.tune_strategy import (
    OneBackendStrategy,
)
from aitune.torch.tune_strategy.tune_strategy import DummyTuneStrategy


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5)  # add fake layer so that model has some parameters

    def forward(self, x, **kwargs):
        return x

    def simulate_redirection(self, x):
        return self.forward(x)

    def simulate_prefill_decode(self, x):
        return self.forward(x, prefill=True) + self.forward(x, prefill=False)


TEST_MODULE_NAME = "demo-identity"
IDENTITY_FORWARD_SIGNATURE = ForwardSignature.from_callable(Identity().forward)


def _identity_metadata(args, kwargs, *, strict=False):
    forward_inputs = IDENTITY_FORWARD_SIGNATURE.normalize(args, kwargs)
    return SampleMetadata.from_inputs(forward_inputs.arguments, strict=strict)


def _assert_sample_stores(mock: Mock, expected_samples: list[list[tuple[tuple, dict]]]) -> list[SampleStore]:
    stores = [mock_call.args[3] for mock_call in mock.call_args_list]
    assert all(isinstance(store, SampleStore) for store in stores)
    assert [list(store) for store in stores] == expected_samples
    return stores


def _torch_inductor_strategy_for_wrapper_tests() -> OneBackendStrategy:
    strategy = OneBackendStrategy(
        backend=TorchInductorJitBackend(),
        profiling_config=ProfilingConfig(
            batch_sizes=[1],
            measuring_strategy=ModelExecutionTimeMeasuringStrategy(),
            measurement_stop_strategy=NumStepsMeasuringStopStrategy(num_steps=1),
            profiling_stop_strategy=AllSamplesProfilingStopStrategy(),
        ),
    )
    strategy.enable_performance_validation(False)
    strategy.enable_find_max_batch_size(False)
    return strategy


@pytest.fixture
def module():
    model = Identity()

    return Module(model, TEST_MODULE_NAME, strategy=DummyTuneStrategy())


def test_init():
    model = Identity()
    module = Module(model, TEST_MODULE_NAME)
    assert module._self_name == TEST_MODULE_NAME
    assert module.state == ModuleState.INIT
    assert module.device == torch.device("cpu")
    assert module._self_wrapper is None
    assert module._self_prev_recording is None
    assert module._self_orig_forward is not None
    assert module._self_proxy_forward is not None


def test_deploy_wrapper_coordinates_deployment(mocker, module):
    wrapper = Mock()
    module._self_wrapper = wrapper
    coordinated = mocker.patch("aitune.torch.module.wrapper_module.coordinator.raise_if_any_rank_fails")
    device = torch.device("cpu")

    module._deploy_wrapper(device)

    coordinated.assert_called_once_with(f"Deploying module {TEST_MODULE_NAME}")
    wrapper.deploy.assert_called_once_with(device=device)


def test_module_registration():
    assert MODULE_REGISTRY.modules == {}

    module = Module(Identity(), TEST_MODULE_NAME)
    assert MODULE_REGISTRY.modules == {TEST_MODULE_NAME: module}


def test_recording(module):
    global_config.strict_mode = True
    module(1)
    assert len(module.graph_specs) == 1
    module(2)
    assert len(module.graph_specs) == 2
    assert module.state == ModuleState.RECORDING


def test_recording_with_functools_wrapped_bound_forward():
    def decorator(forward):
        @functools.wraps(forward)
        def wrapper(self, *args, **kwargs):
            return forward(self, *args, **kwargs)

        return wrapper

    class DecoratedIdentity(Identity):
        @decorator
        def forward(self, x, **kwargs):
            return self.linear(x)

    model = DecoratedIdentity()
    module = Module(model, TEST_MODULE_NAME, strategy=DummyTuneStrategy())
    sample = torch.ones(1, 5)

    output = module(sample, lora_scale=1.0)

    assert torch.equal(output, model.linear(sample))
    assert module.state == ModuleState.RECORDING
    assert len(module.graph_specs) == 1
    assert tuple(parameter.name for parameter in module.graph_specs[0].forward_signature.parameters) == (
        "x",
        "kwargs",
    )


def test_recording_with_update_wrapped_partial_forward():
    model = Identity()

    def context_parallel_forward(module, x, **kwargs):
        assert kwargs["context_parallel"] is True
        return module.linear(x)

    model.forward = functools.update_wrapper(
        functools.partial(context_parallel_forward, model),
        context_parallel_forward,
    )
    module = Module(model, TEST_MODULE_NAME, strategy=DummyTuneStrategy())
    sample = torch.ones(1, 5)

    output = module(sample, context_parallel=True)

    assert torch.equal(output, model.linear(sample))
    assert module.state == ModuleState.RECORDING
    assert len(module.graph_specs) == 1
    assert tuple(parameter.name for parameter in module.graph_specs[0].forward_signature.parameters) == (
        "x",
        "kwargs",
    )


def test_recording_with_explicit_dynamic_shapes():
    dynamic_shapes = {"x": (BatchDim("batch", min=1, opt=2, max=4), 5)}
    module = Module(Identity(), TEST_MODULE_NAME, dynamic_shapes=dynamic_shapes)

    module(torch.ones(2, 5))

    assert module.graph_specs[0].dynamic_shapes == dynamic_shapes


def test_passthrough(module):
    global_config.strict_mode = True
    module(1)
    module.enable_passthrough()
    module(2)
    assert len(module.graph_specs) == 1
    assert module.state == ModuleState.PASSTHROUGH
    module.enable_recording()
    module(3)
    assert len(module.graph_specs) == 2
    assert module.state == ModuleState.RECORDING


def test_forward_hooks(module):
    hook_calls = []

    def pre_hook(module, input):  # noqa: A002
        hook_calls.append("pre_hook")
        return input

    def hook(module, input, output):  # noqa: A002
        hook_calls.append("forward_hook")
        return output

    module.register_forward_hook(hook)
    module.register_forward_pre_hook(pre_hook)

    module(1)
    assert hook_calls == ["pre_hook", "forward_hook"]
    hook_calls.clear()
    module(2)
    assert hook_calls == ["pre_hook", "forward_hook"]
    hook_calls.clear()
    assert module.state == ModuleState.RECORDING
    module.enable_passthrough()
    module(3)
    assert hook_calls == ["pre_hook", "forward_hook"]
    assert module.state == ModuleState.PASSTHROUGH


def test_tune_dry_run(module, torch_device):
    strategy = Mock()
    global_config.strict_mode = True

    with pytest.raises(ValueError, match="Module: 'demo-identity' has not recorded any samples. Cannot tune it."):
        module.tune(strategy=strategy, dry_run=True, device=torch_device)

    # record a sample
    module(1, a=1)
    module(2, b=2)
    module.tune(strategy=strategy, dry_run=True, device=torch_device)
    tuning_device = resolve_tuning_device(torch_device, module.__wrapped__)

    stores = _assert_sample_stores(strategy.tune_dry_run, [[((1,), {"a": 1})], [((2,), {"b": 2})]])

    strategy.tune_dry_run.assert_has_calls([
        call(
            module,
            TEST_MODULE_NAME,
            GraphSpec(
                name="0",
                input_spec=_identity_metadata(args=(1,), kwargs={"a": 1}, strict=True),
                output_spec=SampleMetadata.from_outputs(1, strict=True),
                forward_signature=IDENTITY_FORWARD_SIGNATURE,
                post_input_spec=_identity_metadata(args=(1,), kwargs={"a": 1}, strict=True),
            ),
            stores[0],
            tuning_device,
            aitune_cache_dir() / module._self_name / module.graph_specs[0].name,
        ),
        call(
            module,
            TEST_MODULE_NAME,
            GraphSpec(
                name="1",
                input_spec=_identity_metadata(args=(2,), kwargs={"b": 2}, strict=True),
                output_spec=SampleMetadata.from_outputs(2, strict=True),
                forward_signature=IDENTITY_FORWARD_SIGNATURE,
                post_input_spec=_identity_metadata(args=(2,), kwargs={"b": 2}, strict=True),
            ),
            stores[1],
            tuning_device,
            aitune_cache_dir() / module._self_name / module.graph_specs[1].name,
        ),
    ])

    assert module.state == ModuleState.RECORDING


def test_tune_resolves_unspecified_device_with_module_context(mocker):
    wrapped_module = torch.nn.Identity()
    module = Module(wrapped_module, TEST_MODULE_NAME)
    strategy = Mock()
    module(1)
    resolve_device = mocker.patch(
        "aitune.torch.module.wrapper_module.resolve_tuning_device",
        return_value=torch.device("cuda:1"),
    )

    module.tune(strategy=strategy, dry_run=True)

    resolve_device.assert_called_once_with(None, wrapped_module)


def test_tune(module, torch_device, mocker):
    mock_backend = Mock()
    mock_backend.describe.return_value = "mock backend description"
    strategy = Mock()
    strategy.tune.return_value = mock_backend
    evict_page_cache = mocker.spy(SampleStore, "evict_page_cache")
    global_config.strict_mode = True

    with pytest.raises(ValueError, match="Module: 'demo-identity' has not recorded any samples. Cannot tune it."):
        module.tune(strategy=strategy, dry_run=True, device=torch_device)

    # record a sample
    module(1, a=1)
    module(2, b=2)
    module.tune(strategy=strategy, dry_run=False, device=torch_device)
    tuning_device = resolve_tuning_device(torch_device, module.__wrapped__)

    stores = _assert_sample_stores(strategy.tune, [[((1,), {"a": 1})], [((2,), {"b": 2})]])
    assert [mock_call.args[0] for mock_call in evict_page_cache.call_args_list] == stores

    assert strategy.tune.call_args_list == [
        call(
            module,
            TEST_MODULE_NAME,
            GraphSpec(
                name="0",
                input_spec=_identity_metadata(args=(1,), kwargs={"a": 1}, strict=True),
                output_spec=SampleMetadata.from_outputs(1, strict=True),
                forward_signature=IDENTITY_FORWARD_SIGNATURE,
                post_input_spec=_identity_metadata(args=(1,), kwargs={"a": 1}, strict=True),
            ),
            stores[0],
            tuning_device,
            aitune_cache_dir() / module._self_name / module.graph_specs[0].name,
        ),
        call(
            module,
            TEST_MODULE_NAME,
            GraphSpec(
                name="1",
                input_spec=_identity_metadata(args=(2,), kwargs={"b": 2}, strict=True),
                output_spec=SampleMetadata.from_outputs(2, strict=True),
                forward_signature=IDENTITY_FORWARD_SIGNATURE,
                post_input_spec=_identity_metadata(args=(2,), kwargs={"b": 2}, strict=True),
            ),
            stores[1],
            tuning_device,
            aitune_cache_dir() / module._self_name / module.graph_specs[1].name,
        ),
    ]
    assert module.state == ModuleState.TUNED


def test_tune_evicts_sample_page_cache_on_failure(module, torch_device, mocker):
    strategy = Mock()
    strategy.tune.side_effect = RuntimeError("tuning failed")
    evict_page_cache = mocker.spy(SampleStore, "evict_page_cache")
    module(1)

    with pytest.raises(RuntimeError, match="tuning failed"):
        module.tune(strategy=strategy, device=torch_device)

    evict_page_cache.assert_called_once()


def test_tune_with_provided_strategy(torch_device):
    model = Identity()
    strategy = Mock()
    module = Module(model, TEST_MODULE_NAME, strategy=strategy)

    module(1, a=1)
    module(2, b=2)
    module.tune(dry_run=False, device=torch_device)

    strategy.tune.call_count = 2
    assert module.state == ModuleState.TUNED


def test_tune_with_list_of_strategies(torch_device):
    model = Identity()
    strategy1 = Mock()
    strategy2 = Mock()
    module = Module(model, TEST_MODULE_NAME, strategies=[strategy1, strategy2])

    module(1, a=1)
    module(2, b=2)
    module.tune(dry_run=False, device=torch_device)

    module._self_strategy_list[0].tune.assert_called_once()  # pytype: disable=attribute-error
    module._self_strategy_list[1].tune.assert_called_once()  # pytype: disable=attribute-error
    assert module.state == ModuleState.TUNED

    with pytest.raises(RuntimeError, match="Module is already tuned. Use force=True to reset tuned module."):
        module.enable_recording()

    module.enable_recording(force=True)
    module(1)
    module(2)
    module(3)

    with pytest.raises(
        RuntimeError, match="Not enough strategies for multi-graph. Expected at least 3, got 2."
    ) as exc_info:
        module.tune(dry_run=False, device=torch_device)
    assert "Captured graph specs:" in str(exc_info.value)
    for index, graph_spec in enumerate(module.graph_specs):
        assert f"Graph spec {index}:\n{graph_spec}" in str(exc_info.value)


def test_tune_with_dict_of_strategies(torch_device):
    model = Identity()
    global_config.strict_mode = True
    strategy1 = Mock()
    strategy2 = Mock()
    graph1 = _identity_metadata(args=(1,), kwargs={}, strict=True)
    graph2 = _identity_metadata(args=(2,), kwargs={}, strict=True)
    module = Module(model, TEST_MODULE_NAME, strategies={graph1: strategy1, graph2: strategy2})

    module(1)
    module(2)
    module.tune(dry_run=False, device=torch_device)

    module._self_strategy_map[graph1].tune.assert_called_once()  # pytype: disable=attribute-error
    module._self_strategy_map[graph2].tune.assert_called_once()  # pytype: disable=attribute-error
    assert module.state == ModuleState.TUNED

    with pytest.raises(RuntimeError, match="Module is already tuned. Use force=True to reset tuned module."):
        module.enable_recording()

    module.enable_recording(force=True)
    module(1)
    module(2)
    module(3)

    with pytest.raises(RuntimeError, match=r"The are following errors:\nmissing strategy for graph"):
        module.tune(dry_run=False, device=torch_device)


def test_tune_with_torch_compile_backend_direct_call(module, torch_device):
    module(torch.tensor(1))
    assert len(module.graph_specs) == 1

    strategy = _torch_inductor_strategy_for_wrapper_tests()
    module.tune(strategy=strategy, dry_run=False, device=torch_device)
    assert module.state == ModuleState.TUNED

    module(torch.tensor(1))

    # check that optimized module detects missing backend for a new graph
    with pytest.raises(RuntimeError, match="No backend found for a graph"):
        module(1111)


def test_tune_with_torch_compile_backend_simulate_redirection(module, torch_device):
    module.simulate_redirection(torch.tensor(1))
    assert len(module.graph_specs) == 1

    strategy = _torch_inductor_strategy_for_wrapper_tests()
    module.tune(strategy=strategy, dry_run=False, device=torch_device)
    assert module.state == ModuleState.TUNED

    module.simulate_redirection(torch.tensor(1))

    # check that optimized module detects missing backend for a new graph
    with pytest.raises(RuntimeError, match="No backend found for a graph"):
        module.simulate_redirection(111)


def test_tune_with_torch_compile_backend_simulate_prefill_decode(module, torch_device):
    module.simulate_prefill_decode(torch.tensor(1))
    assert len(module.graph_specs) == 2

    strategy = _torch_inductor_strategy_for_wrapper_tests()
    module.tune(strategy=strategy, dry_run=False, device=torch_device)
    assert module.state == ModuleState.TUNED

    module.simulate_prefill_decode(torch.tensor(1))

    # check that optimized module detects missing backend for a new graph
    with pytest.raises(RuntimeError, match="No backend found for a graph"):
        module.simulate_prefill_decode(111)


def test_serialization(torch_device):
    """Test converting module to dictionary."""
    model = Identity()
    dynamic_shapes = {"x": (BatchDim("batch", min=1, opt=2, max=4), 5)}
    module = Module(model, TEST_MODULE_NAME, dynamic_shapes=dynamic_shapes)

    # Record samples and tune the module
    module(torch.ones(2, 5))
    strategy = _torch_inductor_strategy_for_wrapper_tests()
    module.tune(strategy=strategy, dry_run=False, device=torch_device)
    assert module.state == ModuleState.TUNED

    # Test to_dict
    state_dict = module.to_dict(prefix="test")
    assert isinstance(state_dict, OrderedDict)
    module_data = state_dict["test"]
    assert module_data[Module.TYPE_KEY] == Module.__name__
    assert module_data[Module.NAME_KEY] == TEST_MODULE_NAME
    assert module_data[Module.DYNAMIC_SHAPES_KEY] == dynamic_shapes
    assert Module.TUNED_MODULE_KEY in module_data

    # Test from_dict
    model = Identity()
    module = Module.from_dict(model, state_dict["test"], device=torch_device)
    assert module.state == ModuleState.TUNED
    assert module._self_name == TEST_MODULE_NAME
    assert module._self_dynamic_shapes == dynamic_shapes
    assert isinstance(module._self_wrapper, TunedModule)


def test_to_dict_raises_error_when_not_tuned():
    """Test that to_dict raises error when module is not tuned."""
    module = Module(Identity(), TEST_MODULE_NAME)
    with pytest.raises(RuntimeError, match="Module is not tuned. Cannot save state_dict."):
        module.to_dict()


def test_state_dict_for_passthrough_module_preserves_wrapped_module_state():
    """Passthrough wrappers serialize the original torch module state."""
    inner = Identity()
    expected_state_dict = {name: value.detach().clone() for name, value in inner.state_dict(prefix="wrapped.").items()}
    module = Module(inner, TEST_MODULE_NAME)
    module.enable_passthrough()

    state_dict = module.state_dict(prefix="wrapped.")

    assert list(state_dict.keys()) == list(expected_state_dict.keys())
    for name, value in state_dict.items():
        torch.testing.assert_close(value, expected_state_dict[name])


def test_parent_state_dict_preserves_passthrough_child_module_state():
    """PyTorch recursive state_dict can save a passthrough AITune child wrapper."""

    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = Module(Identity(), "child")
            self.child.enable_passthrough()

    parent = Parent()

    state_dict = parent.state_dict()

    assert list(state_dict.keys()) == ["child.linear.weight", "child.linear.bias"]


def test_from_dict_raises_error_for_invalid_state_dict(torch_device):
    """Test that from_dict raises error for invalid state dict."""
    with pytest.raises(ValueError, match=f"Invalid dictionary format for {Module.__class__.__name__}"):
        Module.from_dict(Identity(), {"type": "invalid"}, device=torch_device)


def test_deepcopy():
    """Test that deepcopy of a wrapped module does not raise an error.

    Wrapper should not add any attributes to the original wrapped model which prevent doing a deepcopy.
    The only exception is overridden forward method - but this gets restored before handing original module to
    a backend.
    """
    model = Identity()
    module = Module(model, TEST_MODULE_NAME)

    module._restore_original_forward()
    deepcopy(model)  # this must not fail


def test_forward_method_should_have_same_signature():
    class TestNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

        def forward(self, x, y, z, pos=4):
            """Test forward method."""
            return x, y, z, pos

    original_model = TestNet()
    expected_keys = {"x", "y", "z", "pos"}

    model = Module(original_model, TEST_MODULE_NAME)
    assert model.state == ModuleState.INIT
    assert set(inspect.signature(original_model.forward).parameters.keys()) == expected_keys
    model(1, 2, 3)
    assert model.state == ModuleState.RECORDING
    assert set(inspect.signature(original_model.forward).parameters.keys()) == expected_keys

    strategy = OneBackendStrategy(TorchEagerBackend()).enable_find_max_batch_size(False)
    model.tune(device=torch.device("cpu"), strategy=strategy)
    assert model.state == ModuleState.TUNED
    assert set(inspect.signature(original_model.forward).parameters.keys()) == expected_keys


# ── capture_outputs hook preservation ────────────────────────────────────────


class _SimpleHookModule(torch.nn.Module):
    """Minimal module used in hook-preservation tests."""

    def forward(self, x):
        return x * 2


def test_external_hook_preserved_in_forward_hooks_after_restore_proxy_cycle():
    """A forward hook registered after wrapping must survive _restore_original_forward/_proxy_forward."""
    inner = _SimpleHookModule()
    wrapper = Module(inner, "hook-test-aot", strategy=DummyTuneStrategy())

    hook_calls = []
    inner.register_forward_hook(lambda mod, inp, out: hook_calls.append(out))

    assert len(inner._forward_hooks) == 1

    wrapper._restore_original_forward()
    assert len(inner._forward_hooks) == 0  # cleared during restore — expected

    wrapper._proxy_forward()
    assert len(inner._forward_hooks) == 1  # must be restored


def test_external_hook_fires_exactly_once_after_restore_proxy_cycle():
    """After the cycle the hook fires exactly once per forward call — no double-firing."""
    inner = _SimpleHookModule()
    wrapper = Module(inner, "hook-fire-aot", strategy=DummyTuneStrategy())

    hook_calls = []
    inner.register_forward_hook(lambda mod, inp, out: hook_calls.append(out))

    wrapper._restore_original_forward()
    wrapper._proxy_forward()

    x = torch.ones(1)
    wrapper(x)  # triggers recording; inner.__call__ fires post-hooks once
    assert len(hook_calls) == 1


def test_external_pre_hook_preserved_after_restore_proxy_cycle():
    """A forward_pre_hook registered after wrapping must also survive the cycle."""
    inner = _SimpleHookModule()
    wrapper = Module(inner, "pre-hook-test-aot", strategy=DummyTuneStrategy())

    pre_calls = []
    inner.register_forward_pre_hook(lambda mod, inp: pre_calls.append(inp))

    assert len(inner._forward_pre_hooks) == 1

    wrapper._restore_original_forward()
    assert len(inner._forward_pre_hooks) == 0

    wrapper._proxy_forward()
    assert len(inner._forward_pre_hooks) == 1


def test_first_proxy_forward_without_prior_restore_does_not_crash():
    """_proxy_forward before any _restore_original_forward must not raise and uses init-time hooks."""
    inner = _SimpleHookModule()
    wrapper = Module(inner, "no-restore-aot", strategy=DummyTuneStrategy())

    wrapper._proxy_forward()  # must not raise
    assert inner._forward_hooks == wrapper._current_forward_hooks


def test_backend_added_hooks():
    """Test that backend added hooks are preserved after a proxy forward."""
    hook_calls = []

    class TestStrategyWhichAddsHooks(TuneStrategy):
        def _tune(
            self,
            module: torch.nn.Module,
            *args,
            **kwargs,
        ):
            # Imitate a backend which adds hooks to the module.
            module.register_forward_pre_hook(lambda mod, inp: hook_calls.append("backend pre hook"))
            module.register_forward_hook(lambda mod, inp, out: hook_calls.append("backend hook"))
            return Mock()

        def describe(self):
            return "Test strategy which adds hooks."

        def _describe_parts(self):
            return ["Test strategy which adds hooks."]

        def to_json_dict(self):
            return {"type": "test_strategy_which_adds_hooks"}

    module = Module(_SimpleHookModule(), "backend-added-hooks-aot", strategy=DummyTuneStrategy())
    module.register_forward_pre_hook(lambda mod, inp: hook_calls.append("module pre hook"))
    module.register_forward_hook(lambda mod, inp, out: hook_calls.append("module hook"))

    module(1)

    module.tune(strategy=TestStrategyWhichAddsHooks(), dry_run=False, device=torch.device("cpu"))
    hook_calls.clear()

    module(1)
    assert hook_calls == ["backend pre hook", "module pre hook", "backend hook", "module hook"]
