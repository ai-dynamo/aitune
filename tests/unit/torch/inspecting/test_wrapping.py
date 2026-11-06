# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn

from aitune.torch.backend import TensorRTBackend, TorchInductorJitBackend
from aitune.torch.inspecting.module_info import ModuleInfo
from aitune.torch.inspecting.module_inspector import DictOfModulesInfo, ListOfModulesInfo
from aitune.torch.inspecting.wrapping import wrap
from aitune.torch.module.forward_signature import ForwardSignature
from aitune.torch.module.wrapper_module import Module
from aitune.torch.tune_strategy import FirstWinsStrategy, OneBackendStrategy
from tests.toy_models import ToyTorchModel
from tests.utilities.helpers import make_input_metadata


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))


@pytest.fixture
def nested_model():
    """Create a nested model for testing."""

    class NestedModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(5, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.linear(x))

    return NestedModule()


def test_wrap_root_module(simple_model):
    """Test wrapping the root module."""
    # When wrapping with root module
    wrapped_model = wrap(
        simple_model,
        [
            ModuleInfo(module=simple_model),
        ],
    )

    # Then the model should be wrapped in a Module
    assert isinstance(wrapped_model, Module)
    assert wrapped_model._self_name == "Sequential"


def test_wrap_submodule(nested_model):
    """Test wrapping a submodule."""
    parent = ModuleInfo(module=nested_model)
    # When wrapping a submodule
    wrapped_model = wrap(
        nested_model,
        [
            ModuleInfo(name="linear", module=nested_model.linear, parent=parent),
        ],
    )

    # Then the submodule should be wrapped
    assert not isinstance(wrapped_model, Module)

    assert isinstance(wrapped_model.linear, Module)
    assert wrapped_model.linear._self_name == "linear"

    # And the original model should be returned
    assert wrapped_model is nested_model


def test_wrap_multiple_modules(nested_model):
    """Test wrapping multiple modules."""
    # When wrapping multiple modules
    parent = ModuleInfo(module=nested_model)

    # When wrapping a submodule
    wrapped_model = wrap(
        nested_model,
        [
            ModuleInfo(name="linear", module=nested_model.linear, parent=parent),
            ModuleInfo(name="relu", module=nested_model.relu, parent=parent),
        ],
    )

    # Then both submodules should be wrapped
    assert isinstance(nested_model.linear, Module)
    assert isinstance(nested_model.relu, Module)
    assert nested_model.linear._self_name == "linear"
    assert nested_model.relu._self_name == "relu"

    # And the original model should be returned
    assert wrapped_model is nested_model


def test_wrap_with_strategy(simple_model):
    """Test wrapping with a strategy."""
    strategy = FirstWinsStrategy(backends=[TorchInductorJitBackend(), TensorRTBackend()])

    # When wrapping with a strategy
    wrapped_model = wrap(
        simple_model,
        [
            ModuleInfo(module=simple_model),
        ],
        strategy=strategy,
    )

    # Then the module should be wrapped with the strategy
    assert isinstance(wrapped_model, Module)
    assert wrapped_model._self_strategy.__class__ == strategy.__class__
    assert wrapped_model._self_strategy_list is None
    assert wrapped_model._self_strategy_map is None


def test_wrap_with_strategies_list(simple_model):
    """Test wrapping with a strategy."""
    strategies = [
        FirstWinsStrategy(backends=[TensorRTBackend(), TorchInductorJitBackend()]),
        OneBackendStrategy(backend=TorchInductorJitBackend()),
    ]

    # When wrapping with a strategy
    wrapped_model = wrap(
        simple_model,
        [
            ModuleInfo(module=simple_model),
        ],
        strategies=strategies,
    )

    # Then the module should be wrapped with the strategy
    assert isinstance(wrapped_model, Module)
    assert wrapped_model._self_strategy is None
    assert wrapped_model._self_strategy_map is None


def test_wrap_with_strategies_map(simple_model):
    """Test wrapping with a strategy."""
    model = ToyTorchModel().to("cpu").eval()
    forward_signature = ForwardSignature.from_callable(model.forward)
    sample = ((model.inputs(device="cpu")[0],), {})
    sample_metadata1 = make_input_metadata(forward_signature, sample)
    sample_metadata2 = make_input_metadata(forward_signature, sample)

    strategies = {
        sample_metadata1: FirstWinsStrategy(backends=[TorchInductorJitBackend(), TensorRTBackend()]),
        sample_metadata2: OneBackendStrategy(backend=TorchInductorJitBackend()),
    }

    # When wrapping with a strategy
    wrapped_model = wrap(
        simple_model,
        [
            ModuleInfo(module=simple_model),
        ],
        strategies=strategies,
    )

    # Then the module should be wrapped with the strategy
    assert isinstance(wrapped_model, Module)
    assert wrapped_model._self_strategy is None
    assert wrapped_model._self_strategy_list is None
    assert set(wrapped_model._self_strategy_map.keys()) == set(strategies.keys())


def test_wrap_empty_modules_list(simple_model):
    """Test wrapping with an empty modules list."""
    # When wrapping with an empty list
    wrapped_model = wrap(simple_model, [])

    # Then the original model should be returned unchanged
    assert not isinstance(wrapped_model, Module)
    assert wrapped_model is simple_model


def test_wrap_module_execution(simple_model):
    """Test that wrapped modules can still be executed."""
    # When wrapping modules
    wrapped_model = wrap(simple_model, [ModuleInfo(module=simple_model)])

    # Then the model should still work
    input_tensor = torch.randn(1, 10)
    output = wrapped_model(input_tensor)
    assert output.shape == (1, 5)


def test_wrap_module_state_preservation(simple_model):
    """Test that module state is preserved after wrapping."""
    # Set some state in the original model
    simple_model[0].weight.data.fill_(1.0)
    original_weight = simple_model[0].weight.clone()

    # When wrapping the module
    wrapped_model = wrap(simple_model, [ModuleInfo(module=simple_model)])

    # Then the state should be preserved
    assert torch.allclose(wrapped_model[0].weight, original_weight)


def test_wrap_custom_object():
    """Test wrapping a custom object with PyTorch modules."""

    class CustomObject:
        def __init__(self):
            self.model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())

        def __call__(self, x):
            return self.model(x)

    custom_obj = CustomObject()
    parent = ModuleInfo(module=custom_obj)

    strategy = FirstWinsStrategy(backends=[TorchInductorJitBackend(), TensorRTBackend()])

    module_info = [ModuleInfo(name="model", module=custom_obj.model, parent=parent)]

    # When wrapping custom object
    wrapped_obj = wrap(custom_obj, module_info, strategy=strategy)

    # Then verify the results
    assert isinstance(wrapped_obj, CustomObject)
    assert isinstance(wrapped_obj.model, Module)


def test_wrap_custom_object_nested():
    """Test wrapping a custom object with nested PyTorch modules."""

    class CustomLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 20)

        def forward(self, x):
            return self.linear(x)

    class CustomObject:
        def __init__(self):
            self.layer = CustomLayer()

        def __call__(self, x):
            return self.layer(x)

    custom_obj = CustomObject()
    parent = ModuleInfo(module=custom_obj)

    strategy = FirstWinsStrategy(backends=[TorchInductorJitBackend(), TensorRTBackend()])

    module_info = [ModuleInfo(name="layer", module=custom_obj.layer, parent=parent)]

    # When wrapping custom object with nested module
    wrapped_obj = wrap(custom_obj, module_info, strategy=strategy)

    # Then verify the results
    assert isinstance(wrapped_obj, CustomObject)
    assert isinstance(wrapped_obj.layer, Module)


def test_wrap_custom_object_multiple_modules():
    """Test wrapping a custom object with multiple PyTorch modules."""

    class CustomObject:
        def __init__(self):
            self.model1 = nn.Linear(10, 20)
            self.model2 = nn.Linear(20, 5)

        def __call__(self, x):
            x = self.model1(x)
            return self.model2(x)

    custom_obj = CustomObject()
    parent = ModuleInfo(module=custom_obj)

    strategy = FirstWinsStrategy(backends=[TorchInductorJitBackend(), TensorRTBackend()])

    module_info = [
        ModuleInfo(name="model1", module=custom_obj.model1, parent=parent),
        ModuleInfo(name="model2", module=custom_obj.model2, parent=parent),
    ]

    # When wrapping multiple modules
    wrapped_obj = wrap(custom_obj, module_info, strategy=strategy)

    # Then verify the results
    assert isinstance(wrapped_obj, CustomObject)
    assert isinstance(wrapped_obj.model1, Module)
    assert isinstance(wrapped_obj.model2, Module)


def test_wrap_custom_object_without_modules():
    """Test wrapping a custom object without PyTorch modules."""

    class CustomObject:
        def __init__(self):
            self.data = torch.randn(5, 5)

        def __call__(self, x):
            return x + self.data

    custom_obj = CustomObject()

    strategy = FirstWinsStrategy(backends=[TorchInductorJitBackend(), TensorRTBackend()])

    # When wrapping custom object without modules
    wrapped_obj = wrap(custom_obj, [], strategy=strategy)

    # Then verify the results
    assert isinstance(wrapped_obj, CustomObject)
    assert not isinstance(wrapped_obj.data, Module)


def test_wrap_custom_object_with_attributes():
    """Test wrapping a custom object with various attribute types."""

    class CustomObject:
        def __init__(self):
            self.model = nn.Linear(10, 20)
            self.data = torch.randn(5, 5)
            self.list_data = [1, 2, 3]
            self.dict_data = {"key": "value"}

        def __call__(self, x):
            return self.model(x)

    custom_obj = CustomObject()
    parent = ModuleInfo(module=custom_obj)

    strategy = FirstWinsStrategy(backends=[TorchInductorJitBackend(), TensorRTBackend()])

    module_info = [ModuleInfo(name="model", module=custom_obj.model, parent=parent)]

    # When wrapping custom object with various attributes
    wrapped_obj = wrap(custom_obj, module_info, strategy=strategy)

    # Then verify the results
    assert isinstance(wrapped_obj, CustomObject)
    assert isinstance(wrapped_obj.model, Module)

    assert not isinstance(wrapped_obj.data, Module)
    assert not isinstance(wrapped_obj.list_data, Module)
    assert not isinstance(wrapped_obj.dict_data, Module)


def test_wrap_module_in_list():
    """Test wrapping a module in a list."""
    model = [nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5)]
    parent = ListOfModulesInfo(module=model)

    wrapped_model = wrap(model, [ModuleInfo(name="0", module=model[0], parent=parent)])
    assert not isinstance(wrapped_model, Module)
    assert isinstance(wrapped_model[0], Module)
    assert not isinstance(wrapped_model[1], Module)
    assert not isinstance(wrapped_model[2], Module)


def test_wrap_module_in_dict():
    """Test wrapping a module in a dict."""
    model = {"linear": nn.Linear(10, 20), "relu": nn.ReLU(), "linear2": nn.Linear(20, 5)}
    parent = DictOfModulesInfo(module=model)

    wrapped_model = wrap(model, [ModuleInfo(name="linear", module=model["linear"], parent=parent)])
    assert not isinstance(wrapped_model, Module)
    assert isinstance(wrapped_model["linear"], Module)
    assert not isinstance(wrapped_model["relu"], Module)
    assert not isinstance(wrapped_model["linear2"], Module)


def test_wrap_nested_diffrent_types():
    class NestedDifferentModules(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 20)
            self.dict = {"linear": nn.Linear(10, 20)}
            self.list = [nn.Linear(10, 20)]

        def forward(self, x):
            return self.relu(self.linear(x))

    model = NestedDifferentModules()
    parent = ModuleInfo(module=model)
    parent_dict = DictOfModulesInfo(module=model.dict, parent=parent)
    parent_list = ListOfModulesInfo(module=model.list, parent=parent)

    modules = [
        ModuleInfo(name="linear", module=model.linear, parent=parent),
        ModuleInfo(name="linear", module=model.dict["linear"], parent=parent_dict),
        ModuleInfo(name="0", module=model.list[0], parent=parent_list),
    ]

    wrapped_model = wrap(model, modules)
    assert not isinstance(wrapped_model, Module)
    assert isinstance(wrapped_model.linear, Module)
    assert isinstance(wrapped_model.dict["linear"], Module)
    assert isinstance(wrapped_model.list[0], Module)


def test_wrap_with_call_tree_keeps_weights_used_outside_the_module():
    from aitune.torch.inspecting import inspect
    from aitune.torch.utils.module import is_kept_on_device
    from tests.toy_models.call_tree_models import VlaToy, vla_inputs

    model = VlaToy().eval()
    tokens, image = vla_inputs()
    with torch.no_grad():
        info = inspect(
            model,
            [(tokens, image)],
            inference_function=lambda t, i: model(t, i),
            number_of_iterations=1,
            warmup_iterations=1,
            min_depth=1,
            max_depth=8,
            call_tree=True,
        )
    head = [m for m in info.get_modules() if m.module is model.lm_head]
    dit = [m for m in info.get_modules() if m.module is model.dit]
    assert head and dit
    embedding, dit_weight = model.llm.embed.weight, next(model.dit.parameters())

    wrap(model, head + dit, call_tree=info.call_tree)

    assert is_kept_on_device(embedding)
    assert not is_kept_on_device(dit_weight)


def _wrap_vla_head():
    from aitune.torch.inspecting import inspect
    from tests.toy_models.call_tree_models import VlaToy, vla_inputs

    model = VlaToy().eval()
    tokens, image = vla_inputs()
    with torch.no_grad():
        info = inspect(
            model,
            [(tokens, image)],
            inference_function=lambda t, i: model(t, i),
            number_of_iterations=1,
            warmup_iterations=1,
            min_depth=1,
            max_depth=8,
            call_tree=True,
        )
    wrap(model, [m for m in info.get_modules() if m.module is model.lm_head], call_tree=info.call_tree)
    return model


def test_a_later_wrap_keeps_the_kept_weights_of_an_earlier_one():
    from aitune.torch.utils.module import is_kept_on_device, offload

    first = _wrap_vla_head()
    second = _wrap_vla_head()

    assert is_kept_on_device(first.llm.embed.weight)
    assert is_kept_on_device(second.llm.embed.weight)

    offload(first.llm, "meta")  # what tuning the first model does to a module that holds the tied weight
    assert first.llm.embed.weight.device.type == "cpu"
    assert all(p.device.type == "meta" for n, p in first.llm.named_parameters() if n != "embed.weight")


def test_wrapping_a_module_again_replaces_what_it_kept():
    from aitune.torch.utils.module import is_kept_on_device

    model = _wrap_vla_head()
    other = _wrap_vla_head()

    head = ModuleInfo(name="lm_head", module=model.lm_head, parent=ModuleInfo(name="", module=model))
    wrap(model, [head])  # without a call tree: lm_head now keeps nothing

    assert not is_kept_on_device(model.llm.embed.weight)
    assert is_kept_on_device(other.llm.embed.weight)


def test_a_tensor_kept_by_one_wrapped_module_stays_kept_when_another_holder_is_wrapped():
    """Two pipelines share one model: wrapping its embedding for the second keeps the first's tied weight."""
    from aitune.torch.inspecting import inspect
    from aitune.torch.utils.module import is_kept_on_device

    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 8)
            self.head = torch.nn.Linear(8, 16, bias=False)
            self.head.weight = self.embed.weight

        def forward(self, ids):
            return self.head(self.embed(ids))

    shared = Tied()
    ids = torch.randint(0, 16, (2, 4))
    with torch.no_grad():
        first = inspect(
            shared, [ids], inference_function=shared, number_of_iterations=1, warmup_iterations=1, call_tree=True
        )
        second = inspect(
            shared, [ids], inference_function=shared.embed, number_of_iterations=1, warmup_iterations=1, call_tree=True
        )

    wrap(
        shared,
        [ModuleInfo(name="head", module=shared.head, parent=ModuleInfo(name="", module=shared))],
        call_tree=first.call_tree,
    )
    wrap(
        shared,
        [ModuleInfo(name="embed", module=shared.embed, parent=ModuleInfo(name="", module=shared))],
        call_tree=second.call_tree,
    )

    assert is_kept_on_device(shared.embed.weight)


def test_a_later_wrap_with_a_shared_pipeline_call_tree_keeps_the_earlier_models_weights():
    from aitune.torch.inspecting import inspect
    from aitune.torch.utils.module import is_kept_on_device

    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 8)
            self.head = torch.nn.Linear(8, 16, bias=False)
            self.head.weight = self.embed.weight

        def forward(self, ids):
            return self.head(self.embed(ids))

    class Pipeline:
        def __init__(self):
            self.a, self.b = Tied(), Tied()

        def __call__(self, ids):
            return self.a(ids) + self.b(ids)

    pipe = Pipeline()
    ids = torch.randint(0, 16, (2, 4))
    with torch.no_grad():
        info = inspect(
            pipe, [ids], inference_function=pipe, number_of_iterations=1, warmup_iterations=1, call_tree=True
        )

    for name, model in (("a", pipe.a), ("b", pipe.b)):
        head = ModuleInfo(name="head", module=model.head, parent=ModuleInfo(name=name, module=model))
        wrap(pipe, [head], call_tree=info.call_tree)

    assert is_kept_on_device(pipe.a.embed.weight)
    assert is_kept_on_device(pipe.b.embed.weight)


def test_wrap_without_call_tree_keeps_nothing(nested_model):
    from aitune.torch.utils.module import is_kept_on_device

    module_info = ModuleInfo(name="linear", module=nested_model.linear, parent=ModuleInfo(name="", module=nested_model))
    wrap(nested_model, [module_info])

    assert not any(is_kept_on_device(p) for p in nested_model.parameters())


def test_wrap_with_call_tree_keeps_buffers_of_a_module_moved_from_outside():
    from aitune.torch.inspecting import inspect
    from aitune.torch.utils.module import is_kept_on_device
    from tests.toy_models.call_tree_models import SequentialSubclassToy

    model = SequentialSubclassToy().eval()
    x = torch.randn(2, 8)
    with torch.no_grad():
        info = inspect(
            model,
            x,
            inference_function=model.encode,
            number_of_iterations=1,
            warmup_iterations=1,
            min_depth=1,
            max_depth=8,
            call_tree=True,
        )
    selected = [m for m in info.get_modules() if m.module is model[0]]
    assert selected

    wrap(model, selected, call_tree=info.call_tree)

    assert all(is_kept_on_device(b) for b in model[0].buffers())
    assert all(is_kept_on_device(p) for p in model[0].parameters())


def test_wrap_with_call_tree_checks_a_module_whose_forward_takes_a_callable(caplog):
    from aitune.torch.inspecting import inspect

    class Act(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x, act=torch.tanh):
            return act(self.helper(x))

        def helper(self, x):
            return self.lin(x)

    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = Act()

        def forward(self, x):
            return self.act(x, act=torch.tanh)

    model = Root()
    with torch.no_grad():
        info = inspect(
            model, torch.randn(2, 4), number_of_iterations=1, warmup_iterations=1, min_depth=1, call_tree=True
        )
    selected = [m for m in info.get_modules() if m.module is model.act]
    assert selected

    wrap(model, selected, call_tree=info.call_tree)  # does not raise; nothing is shared

    assert isinstance(model.act, Module)
