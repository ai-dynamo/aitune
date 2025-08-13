# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific
"""Test for wrapper module."""

from collections import OrderedDict
from copy import deepcopy
from unittest.mock import Mock, call

import pytest
import torch

from aitune.torch.backend.torch_inductor_backend import TorchInductorBackend
from aitune.torch.config import aitune_cache_dir
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.module.tuned_module import TunedModule
from aitune.torch.module.wrapper_module import Module, ModuleState
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_strategy import (
    FirstWinsStrategy,
    HighestThroughputStrategy,
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


def test_module_registration():
    assert MODULE_REGISTRY.modules == {}

    module = Module(Identity(), TEST_MODULE_NAME)
    assert MODULE_REGISTRY.modules == {TEST_MODULE_NAME: module}


def test_recording(module):
    module(1)
    assert len(module.graph_specs) == 1
    module(2)
    assert len(module.graph_specs) == 2
    assert module.state == ModuleState.RECORDING


def test_passthrough(module):
    module(1)
    module.enable_passthrough()
    module(2)
    assert len(module.graph_specs) == 1
    assert module.state == ModuleState.PASSTHROUGH
    module.enable_recording()
    module(3)
    assert len(module.graph_specs) == 2
    assert module.state == ModuleState.RECORDING


def test_tune_dry_run(module, torch_device):
    strategy = Mock()

    with pytest.raises(ValueError, match="Module: 'demo-identity' has not recorded any samples. Cannot tune it."):
        module.tune(strategy=strategy, dry_run=True, device=torch_device)

    # record a sample
    module(1, a=1)
    module(2, b=2)
    module.tune(strategy=strategy, dry_run=True, device=torch_device)

    strategy.tune_dry_run.assert_has_calls([
        call(
            module,
            TEST_MODULE_NAME,
            GraphSpec(
                name="0",
                input_spec=SampleMetadata.from_sample(((1,), {"a": 1}), prefix="input"),
                output_spec=SampleMetadata.from_sample(1, prefix="output"),
            ),
            [
                ((1,), {"a": 1}),
            ],
            torch_device,
            aitune_cache_dir() / module._self_name / module.graph_specs[0].name,
        ),
        call(
            module,
            TEST_MODULE_NAME,
            GraphSpec(
                name="1",
                input_spec=SampleMetadata.from_sample(((2,), {"b": 2}), prefix="input"),
                output_spec=SampleMetadata.from_sample(2, prefix="output"),
            ),
            [((2,), {"b": 2})],
            torch_device,
            aitune_cache_dir() / module._self_name / module.graph_specs[1].name,
        ),
    ])

    assert module.state == ModuleState.RECORDING


def test_tune(module, torch_device):
    strategy = Mock()

    with pytest.raises(ValueError, match="Module: 'demo-identity' has not recorded any samples. Cannot tune it."):
        module.tune(strategy=strategy, dry_run=True, device=torch_device)

    # record a sample
    module(1, a=1)
    module(2, b=2)
    module.tune(strategy=strategy, dry_run=False, device=torch_device)

    strategy.tune.assert_has_calls([
        call(
            module,
            TEST_MODULE_NAME,
            GraphSpec(
                name="0",
                input_spec=SampleMetadata.from_sample(((1,), {"a": 1}), prefix="input"),
                output_spec=SampleMetadata.from_sample(1, prefix="output"),
            ),
            [((1,), {"a": 1})],
            torch_device,
            aitune_cache_dir() / module._self_name / module.graph_specs[0].name,
        ),
        call(
            module,
            TEST_MODULE_NAME,
            GraphSpec(
                name="1",
                input_spec=SampleMetadata.from_sample(((2,), {"b": 2}), prefix="input"),
                output_spec=SampleMetadata.from_sample(2, prefix="output"),
            ),
            [((2,), {"b": 2})],
            torch_device,
            aitune_cache_dir() / module._self_name / module.graph_specs[1].name,
        ),
    ])

    assert module.state == ModuleState.TUNED


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

    with pytest.raises(RuntimeError, match="Not enough strategies for multi-graph. Expected at least 3, got 2."):
        module.tune(dry_run=False, device=torch_device)


def test_tune_with_dict_of_strategies(torch_device):
    model = Identity()
    strategy1 = Mock()
    strategy2 = Mock()
    graph1 = SampleMetadata.from_sample(((1,), {}), prefix="irrelevant")
    graph2 = SampleMetadata.from_sample(((2,), {}), prefix="irrelevant")
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

    with pytest.raises(RuntimeError, match=r"The are following errors:\nmissing strategy for graph:\(\(3,\), \{\}\)"):
        module.tune(dry_run=False, device=torch_device)


def test_tune_with_torch_compile_backend_direct_call(module, torch_device):
    module(torch.tensor(1))
    assert len(module.graph_specs) == 1

    strategy = OneBackendStrategy(backend=TorchInductorBackend())
    module.tune(strategy=strategy, dry_run=False, device=torch_device)
    assert module.state == ModuleState.TUNED

    module(torch.tensor(1))

    # check that optimized module detects missing backend for a new graph
    with pytest.raises(RuntimeError, match="No backend found for a graph"):
        module(1111)


def test_tune_with_torch_compile_backend_simulate_redirection(module, torch_device):
    module.simulate_redirection(torch.tensor(1))
    assert len(module.graph_specs) == 1

    strategy = OneBackendStrategy(backend=TorchInductorBackend())
    module.tune(strategy=strategy, dry_run=False, device=torch_device)
    assert module.state == ModuleState.TUNED

    module.simulate_redirection(torch.tensor(1))

    # check that optimized module detects missing backend for a new graph
    with pytest.raises(RuntimeError, match="No backend found for a graph"):
        module.simulate_redirection(111)


def test_tune_with_torch_compile_backend_simulate_prefill_decode(module, torch_device):
    module.simulate_prefill_decode(torch.tensor(1))
    assert len(module.graph_specs) == 2

    strategy = OneBackendStrategy(backend=TorchInductorBackend()).enable_find_max_batch_size(False)
    module.tune(strategy=strategy, dry_run=False, device=torch_device)
    assert module.state == ModuleState.TUNED

    module.simulate_prefill_decode(torch.tensor(1))

    # check that optimized module detects missing backend for a new graph
    with pytest.raises(RuntimeError, match="No backend found for a graph"):
        module.simulate_prefill_decode(111)


def test_serialization(torch_device):
    """Test converting module to dictionary."""
    model = Identity()
    module = Module(model, TEST_MODULE_NAME)

    # Record samples and tune the module
    module(torch.tensor(1))
    strategy = OneBackendStrategy(backend=TorchInductorBackend())
    module.tune(strategy=strategy, dry_run=False, device=torch_device)
    assert module.state == ModuleState.TUNED

    # Test to_dict
    state_dict = module.to_dict(prefix="test")
    assert isinstance(state_dict, OrderedDict)
    module_data = state_dict["test"]
    assert module_data[Module.TYPE_KEY] == Module.__name__
    assert module_data[Module.NAME_KEY] == TEST_MODULE_NAME
    assert Module.TUNED_MODULE_KEY in module_data

    # Test from_dict
    model = Identity()
    module = Module.from_dict(model, state_dict["test"], device=torch_device)
    assert module.state == ModuleState.TUNED
    assert module._self_name == TEST_MODULE_NAME
    assert isinstance(module._self_wrapper, TunedModule)


def test_to_dict_raises_error_when_not_tuned():
    """Test that to_dict raises error when module is not tuned."""
    module = Module(Identity(), TEST_MODULE_NAME)
    with pytest.raises(RuntimeError, match="Module is not tuned. Cannot save state_dict."):
        module.to_dict()


def test_from_dict_raises_error_for_invalid_state_dict(torch_device):
    """Test that from_dict raises error for invalid state dict."""
    with pytest.raises(ValueError, match=f"Invalid dictionary format for {Module.__class__.__name__}"):
        Module.from_dict(Identity(), {"type": "invalid"}, device=torch_device)


@pytest.mark.parametrize(
    "strategy_class,strategy_args",
    [
        (OneBackendStrategy, {"backend": TorchInductorBackend()}),
        (FirstWinsStrategy, {"backends": [TorchInductorBackend()]}),
        (HighestThroughputStrategy, {"backends": [TorchInductorBackend()]}),
    ],
)
def test_deepcopy(strategy_class, strategy_args):
    """Test that deepcopy of a wrapped module does not raise an error.

    Wrapper should not add any attributes to the wrapped model. The following test checks that module can be deepcopied.

    Args:
        strategy_class: The strategy class to test
        strategy_args: Arguments to pass to the strategy constructor
    """
    model = Identity()
    strategy = strategy_class(**strategy_args)
    _ = Module(model, TEST_MODULE_NAME, strategy=strategy)
    deepcopy(model)
