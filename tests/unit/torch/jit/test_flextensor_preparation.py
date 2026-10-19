# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preparation contracts using only the public helper and a public-surface FT fake."""

import copy
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import wrapt

from aitune.torch.jit.flextensor import FlexTensorUnit, PreparedFlexTensor, prepare_flextensor_units


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)
        self.norm = torch.nn.LayerNorm(2)
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.register_buffer("offset", torch.zeros(2))

    def forward(self, value):
        return self.norm(self.projection(value)) * self.scale + self.offset


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block(), Block()])
        self.head = torch.nn.Linear(2, 2)
        self.root_scale = torch.nn.Parameter(torch.ones(()))
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        for layer in self.layers:
            value = layer(value)
        return self.head(value) * self.root_scale


class PublicConfig:
    def __init__(self, **kwargs):
        self.include_patterns = ["layers.*"]
        self.exclude_patterns = []
        self.external_compile = True
        self.profile_mode = "view"
        self.max_gpu_mem_fraction = 0.3
        self.pinned_memory_mode = "host_register"
        self.profile_storage_dir = "profile-directory-is-not-an-implicit-load"
        self.strategy = object()
        self.copy_calls = []
        vars(self).update(kwargs)

    def model_copy(self, *, update=None, deep=False):
        assert deep is False, "Arbitrary application strategy objects must not be deep-copied"
        assert update is not None
        assert set(update) <= {"include_patterns", "exclude_patterns"}
        assert all(isinstance(value, list) for value in update.values())
        assert all(isinstance(item, str) and item.strip() for value in update.values() for item in value)
        self.copy_calls.append((update, deep))
        result = copy.copy(self)
        vars(result).update(update)
        return result


class CallableProxy(wrapt.ObjectProxy):
    def __call__(self, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)


class PublicProvider(ModuleType):
    OffloadConfig = PublicConfig

    def __init__(self):
        super().__init__("flextensor")
        self.managers = {}
        self.calls = []
        self.proxy = None
        self.error = None

    def get_offload_manager(self, name="default"):
        if name not in self.managers:
            self.managers[name] = SimpleNamespace(phase=SimpleNamespace(value="not_initialized"), model=None)
        return self.managers[name]

    def offload(self, model, *, config, name="default", compile_fn=None):
        return self._offload("offload", model, config, name, compile_fn, None)

    def offload_from_profile(self, model, profile_path, *, config, name="default", compile_fn=None):
        return self._offload("offload_from_profile", model, config, name, compile_fn, profile_path)

    def _offload(self, operation, model, config, name, compile_fn, profile_path):
        self.calls.append(
            SimpleNamespace(
                operation=operation,
                model=model,
                config=config,
                name=name,
                compile_fn=compile_fn,
                profile_path=profile_path,
            )
        )
        if self.error is not None:
            raise self.error
        manager = self.get_offload_manager(name)
        manager.model = model
        manager.phase.value = "discovery"
        self.proxy = CallableProxy(model)
        return self.proxy

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise AssertionError(f"Helper accessed an unsupported provider API: {name}")


@pytest.fixture
def provider(monkeypatch):
    provider = PublicProvider()
    monkeypatch.setitem(sys.modules, "flextensor", provider)
    return provider


def prepare(model, provider, **kwargs):
    config = kwargs.pop("ft_config", provider.OffloadConfig())
    return prepare_flextensor_units(model, ft_config=config, **kwargs)


def assert_unprepared(model, original_layers):
    assert tuple(model.layers) == original_layers
    assert not any(isinstance(module, FlexTensorUnit) for module in model.modules())


def test_root_form_returns_exact_provider_proxy_and_public_metadata(provider):
    model = Model()
    layers = tuple(model.layers)
    config = provider.OffloadConfig()

    result = prepare_flextensor_units(model, ft_config=config, name="transformer")

    assert isinstance(result, PreparedFlexTensor)
    assert result.model is provider.proxy
    assert result.model is not model
    assert result.manager is provider.get_offload_manager("transformer")
    assert result.effective_config is provider.calls[0].config
    assert result.effective_config is not config
    assert result.unit_paths == ("layers.0", "layers.1")
    assert all(shell.compute is original for shell, original in zip(model.layers, layers, strict=True))
    assert provider.calls[0].model is model
    assert provider.calls[0].name == "transformer"
    assert model.calls == 0


@pytest.mark.parametrize("owner_kind", ["namespace", "module"])
def test_owner_form_installs_exact_proxy_and_preserves_sibling_attributes(provider, owner_kind):
    model = Model()
    owner = SimpleNamespace() if owner_kind == "namespace" else torch.nn.Module()
    owner.transformer = model
    sibling = object()
    owner.sibling = sibling

    result = prepare(owner, provider, target="transformer")

    assert owner.transformer is provider.proxy is result.model
    assert owner.sibling is sibling
    assert result.manager is provider.get_offload_manager()
    assert model.calls == 0


def test_dotted_target_uses_explicit_owner_and_does_not_rebind_other_aliases(provider):
    model = Model()
    owner = SimpleNamespace(pipeline=SimpleNamespace(transformer=model), other_alias=model)

    result = prepare(owner, provider, target="pipeline.transformer")

    assert owner.pipeline.transformer is result.model
    assert owner.other_alias is model
    assert owner.other_alias is not result.model


def test_prepared_result_is_immutable(provider):
    result = prepare(Model(), provider)

    with pytest.raises((AttributeError, TypeError)):
        result.model = object()


def test_validated_shallow_copy_retains_unrelated_policy_and_source_config(provider):
    config = provider.OffloadConfig(exclude_patterns=["layers.0.norm"])
    original_includes = config.include_patterns
    original_excludes = config.exclude_patterns
    original_strategy = config.strategy

    result = prepare(Model(), provider, ft_config=config)

    assert config.include_patterns is original_includes
    assert config.exclude_patterns is original_excludes
    assert config.include_patterns == ["layers.*"]
    assert config.exclude_patterns == ["layers.0.norm"]
    assert len(config.copy_calls) >= 1
    assert result.effective_config.strategy is original_strategy
    assert result.effective_config.max_gpu_mem_fraction == config.max_gpu_mem_fraction
    assert result.effective_config.pinned_memory_mode == config.pinned_memory_mode
    assert result.effective_config.profile_storage_dir == config.profile_storage_dir


@pytest.mark.parametrize("external_compile", [False, True])
def test_helper_preserves_the_two_explicit_compile_modes(provider, external_compile):
    config = provider.OffloadConfig(external_compile=external_compile, profile_mode="view")
    result = prepare(Model(), provider, ft_config=config)
    callback = provider.calls[0].compile_fn

    assert result.effective_config.external_compile is external_compile
    if external_compile:
        assert callback is None
    else:
        assert callable(callback)
        untouched = torch.nn.Linear(2, 2)
        assert callback(untouched) is untouched


@pytest.mark.parametrize("profile_mode", ["getter", "unsupported"])
def test_unqualified_retained_mode_fails_before_mutation(provider, profile_mode):
    model = Model()
    original = tuple(model.layers)
    config = provider.OffloadConfig(external_compile=False, profile_mode=profile_mode)

    with pytest.raises((ValueError, RuntimeError), match="view|mode"):
        prepare(model, provider, ft_config=config)

    assert_unprepared(model, original)
    assert provider.calls == []


def test_profile_loading_is_explicit_and_passes_path_name_and_exact_config(provider, tmp_path):
    profile_path = tmp_path / "saved profile"
    result = prepare(Model(), provider, name="saved", load_profile_from=profile_path)
    call = provider.calls[0]

    assert call.operation == "offload_from_profile"
    assert call.profile_path == str(profile_path)
    assert call.name == "saved"
    assert call.config is result.effective_config


def test_profile_storage_directory_does_not_implicitly_load_profile(provider):
    prepare(Model(), provider)

    assert provider.calls[0].operation == "offload"
    assert provider.calls[0].profile_path is None


def test_preparation_preserves_parameter_buffer_hook_and_mixed_training_identity(provider):
    model = Model()
    model.layers[0].eval()
    original = tuple(model.layers)
    parameters = tuple(model.parameters())
    buffers = tuple(model.buffers())
    pointers = tuple(tensor.data_ptr() for tensor in (*parameters, *buffers))
    original_training = {id(module): module.training for module in model.modules()}
    hooks = []
    original[0].register_forward_pre_hook(lambda *args: hooks.append("pre"))
    original[0].register_forward_hook(lambda *args: hooks.append("post"))

    result = prepare(model, provider)

    assert tuple(model.parameters()) == parameters
    assert tuple(model.buffers()) == buffers
    assert tuple(tensor.data_ptr() for tensor in (*model.parameters(), *model.buffers())) == pointers
    assert all(
        module.training == original_training[id(module)]
        for module in model.modules()
        if id(module) in original_training
    )
    assert tuple(shell.training for shell in model.layers) == (False, True)
    assert hooks == []
    assert model.calls == 0
    with torch.no_grad():
        result.model(torch.ones(1, 2))
    assert hooks == ["pre", "post"]


@pytest.mark.parametrize("invalid", [None, object(), {"include_patterns": ["layers.*"]}])
def test_invalid_config_fails_before_model_mutation(provider, invalid):
    model = Model()
    original = tuple(model.layers)

    with pytest.raises((TypeError, ValueError)):
        prepare(model, provider, ft_config=invalid)

    assert_unprepared(model, original)
    assert provider.calls == []


@pytest.mark.parametrize("invalid", [None, object(), torch.ones(2)])
def test_model_must_be_a_torch_module(provider, invalid):
    with pytest.raises((TypeError, ValueError)):
        prepare(invalid, provider)
    assert provider.calls == []


@pytest.mark.parametrize("invalid", ["", " ", 7, None])
def test_invalid_manager_name_fails_before_mutation(provider, invalid):
    model = Model()
    original = tuple(model.layers)

    with pytest.raises((TypeError, ValueError)):
        prepare(model, provider, name=invalid)

    assert_unprepared(model, original)
    assert provider.calls == []


@pytest.mark.parametrize("invalid", ["", ".transformer", "transformer.", "pipeline..transformer", 7])
def test_invalid_target_fails_before_mutation(provider, invalid):
    model = Model()
    original = tuple(model.layers)

    with pytest.raises((TypeError, ValueError, AttributeError)):
        prepare(SimpleNamespace(transformer=model), provider, target=invalid)

    assert_unprepared(model, original)
    assert provider.calls == []


def test_missing_target_fails_before_any_offload(provider):
    with pytest.raises((AttributeError, ValueError)):
        prepare(SimpleNamespace(), provider, target="transformer")
    assert provider.calls == []


def test_target_must_resolve_to_module(provider):
    with pytest.raises((TypeError, ValueError)):
        prepare(SimpleNamespace(transformer=object()), provider, target="transformer")
    assert provider.calls == []


def test_read_only_property_target_is_rejected_before_model_mutation(provider):
    class Owner:
        @property
        def transformer(self):
            return model

    model = Model()
    original = tuple(model.layers)

    with pytest.raises((AttributeError, ValueError, TypeError)):
        prepare(Owner(), provider, target="transformer")

    assert_unprepared(model, original)
    assert provider.calls == []


@pytest.mark.parametrize("active_state", ["model", "discovery", "profiling", "inference"])
def test_active_named_manager_fails_before_model_mutation(provider, active_state):
    manager = provider.get_offload_manager("existing")
    if active_state == "model":
        manager.model = torch.nn.Linear(2, 2)
    else:
        manager.phase.value = active_state
    model = Model()
    original = tuple(model.layers)

    with pytest.raises((ValueError, RuntimeError), match="fresh|manager|initialized"):
        prepare(model, provider, name="existing")

    assert_unprepared(model, original)
    assert provider.calls == []


def test_named_manager_cannot_be_reused_with_another_model(provider):
    prepare(Model(), provider)
    second = Model()
    original = tuple(second.layers)

    with pytest.raises((ValueError, RuntimeError)):
        prepare(second, provider)

    assert_unprepared(second, original)
    assert len(provider.calls) == 1


def test_provider_failure_propagates_without_claiming_rollback(provider):
    error = RuntimeError("provider setup failed")
    provider.error = error

    with pytest.raises(RuntimeError, match="provider setup failed") as raised:
        prepare(Model(), provider)

    assert raised.value is error
    assert len(provider.calls) == 1


def test_custom_owner_assignment_errors_propagate_without_bypassing_setter(provider):
    class Owner:
        def __init__(self, model):
            self.value = model

        @property
        def transformer(self):
            return self.value

        @transformer.setter
        def transformer(self, value):
            raise RuntimeError("application refuses transformer replacement")

    model = Model()
    owner = Owner(model)

    with pytest.raises(RuntimeError, match="application refuses transformer replacement"):
        prepare(owner, provider, target="transformer")

    assert len(provider.calls) == 1
    assert owner.transformer is model
    assert model.calls == 0


@pytest.mark.parametrize("invalid", ["", b"bytes-profile", 7])
def test_invalid_profile_path_fails_before_model_mutation(provider, invalid):
    model = Model()
    original = tuple(model.layers)

    with pytest.raises((TypeError, ValueError)):
        prepare(model, provider, load_profile_from=invalid)

    assert_unprepared(model, original)
    assert provider.calls == []


@pytest.mark.parametrize(
    "patterns",
    [
        ["layers.*"],
        ["name:layers.*"],
        ["class:Block"],
        [f"class:{Block.__module__}.{Block.__qualname__}"],
        ["class:*Block"],
    ],
)
def test_original_name_and_class_selectors_choose_same_ordered_blocks(provider, patterns):
    config = provider.OffloadConfig(include_patterns=patterns)

    result = prepare(Model(), provider, ft_config=config)

    assert result.unit_paths == ("layers.0", "layers.1")
    assert result.effective_config.include_patterns[: len(patterns)] == patterns
    assert {"name:layers.0", "name:layers.1"} <= set(result.effective_config.include_patterns)


def test_class_exclusion_inside_unit_keeps_only_original_descendant_parameters_resident(provider):
    config = provider.OffloadConfig(include_patterns=["class:Block"], exclude_patterns=["class:LayerNorm"])

    result = prepare(Model(), provider, ft_config=config)

    assert result.unit_paths == ("layers.0", "layers.1")
    assert set(result.effective_config.exclude_patterns) == {
        "name:root_scale",
        "name:head.weight",
        "name:head.bias",
        "name:layers.0.compute.norm.weight",
        "name:layers.0.compute.norm.bias",
        "name:layers.1.compute.norm.weight",
        "name:layers.1.compute.norm.bias",
    }


def test_parameter_only_include_keeps_biases_and_unmatched_descendants_resident(provider):
    config = provider.OffloadConfig(include_patterns=["layers.*.projection.weight"])

    result = prepare(Model(), provider, ft_config=config)

    assert result.unit_paths == ("layers.0.projection", "layers.1.projection")
    assert set(result.effective_config.exclude_patterns) == {
        "name:root_scale",
        "name:head.weight",
        "name:head.bias",
        "name:layers.0.scale",
        "name:layers.0.projection.compute.bias",
        "name:layers.0.norm.weight",
        "name:layers.0.norm.bias",
        "name:layers.1.scale",
        "name:layers.1.projection.compute.bias",
        "name:layers.1.norm.weight",
        "name:layers.1.norm.bias",
    }


def test_recursive_parameter_star_selects_deep_weights_without_selecting_containers(provider):
    result = prepare(Model(), provider, ft_config=provider.OffloadConfig(include_patterns=["*.weight"]))

    assert result.unit_paths == ("layers.0.projection", "layers.0.norm", "layers.1.projection", "layers.1.norm", "head")
    assert "name:layers.0.projection.compute.bias" in result.effective_config.exclude_patterns
    assert "name:layers.0.projection.compute.weight" not in result.effective_config.exclude_patterns


def test_global_mask_preserves_explicitly_included_root_parameter(provider):
    config = provider.OffloadConfig(include_patterns=["layers.*", "root_scale"])

    result = prepare(Model(), provider, ft_config=config)

    assert result.unit_paths == ("layers.0", "layers.1")
    assert "root_scale" in result.effective_config.include_patterns
    assert "name:root_scale" not in result.effective_config.exclude_patterns
    assert set(result.effective_config.exclude_patterns) == {"name:head.weight", "name:head.bias"}


def test_eligible_direct_parameters_outside_units_do_not_create_new_boundaries(provider):
    model = torch.nn.Module()
    model.a = Block()
    model.outside = torch.nn.Linear(2, 2)
    result = prepare(model, provider, ft_config=provider.OffloadConfig(include_patterns=["a.*"]))

    assert result.unit_paths == ("a.projection", "a.norm")
    assert "a.*" in result.effective_config.include_patterns
    assert "name:a.scale" not in result.effective_config.exclude_patterns
    assert "name:a.scale" not in result.effective_config.include_patterns
    assert set(result.effective_config.exclude_patterns) == {"name:outside.weight", "name:outside.bias"}


def test_excluded_parent_still_suppresses_independently_matching_descendant(provider):
    config = provider.OffloadConfig(
        include_patterns=["layers.*", "layers.*.projection", "head"], exclude_patterns=["layers.0"]
    )
    model = Model()
    original_dropped = model.layers[0]

    result = prepare(model, provider, ft_config=config)

    assert result.unit_paths == ("layers.1", "head")
    assert model.layers[0] is original_dropped
    assert not isinstance(model.layers[0].projection, FlexTensorUnit)
    assert "name:layers.0" in result.effective_config.include_patterns
    assert "name:layers.0" in result.effective_config.exclude_patterns
    assert "name:layers.1" in result.effective_config.include_patterns


def test_mixed_nested_parameter_excludes_are_mapped_without_broadening(provider):
    config = provider.OffloadConfig(exclude_patterns=["layers.0.norm", "layers.*.projection.bias"])

    result = prepare(Model(), provider, ft_config=config)

    assert set(result.effective_config.exclude_patterns) == {
        "name:root_scale",
        "name:head.weight",
        "name:head.bias",
        "name:layers.0.compute.norm.weight",
        "name:layers.0.compute.norm.bias",
        "name:layers.0.compute.projection.bias",
        "name:layers.1.compute.projection.bias",
    }


def test_existing_compute_child_is_not_mistaken_for_an_inserted_shell(provider):
    class OriginalComputeOwner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.compute = torch.nn.Linear(2, 2)

        def forward(self, value):
            return self.compute(value)

    model = torch.nn.Module()
    model.block = OriginalComputeOwner()
    original = model.block
    child = original.compute

    result = prepare(model, provider, ft_config=provider.OffloadConfig(include_patterns=["block"]))

    assert result.unit_paths == ("block",)
    assert model.block.compute is original
    assert model.block.compute.compute is child


@pytest.mark.parametrize(
    "pattern, expected",
    [(["layer?"], ("layer0", "layer1")), (["layer[0]"], ("layer[0]",)), (["name:class:literal"], ("class:literal",))],
)
def test_question_mark_brackets_and_explicit_name_prefix_follow_ft_rules(provider, pattern, expected):
    model = torch.nn.Module()
    for name in ("layer0", "layer1", "layer[0]", "class:literal"):
        model.add_module(name, torch.nn.Linear(2, 2))

    result = prepare(model, provider, ft_config=provider.OffloadConfig(include_patterns=pattern))

    assert result.unit_paths == expected


@pytest.mark.parametrize("patterns", [[], ["does_not_exist"], ["layers.*.does_not_exist"]])
def test_empty_or_unmatched_selection_fails_without_wrapping(provider, patterns):
    model = Model()
    original = tuple(model.layers)

    with pytest.raises((ValueError, RuntimeError), match="select|unit|match"):
        prepare(model, provider, ft_config=provider.OffloadConfig(include_patterns=patterns))

    assert_unprepared(model, original)
    assert provider.calls == []


@pytest.mark.parametrize(
    "container",
    [
        torch.nn.ModuleList([torch.nn.Linear(2, 2)]),
        torch.nn.ModuleDict({"linear": torch.nn.Linear(2, 2)}),
        torch.nn.ParameterList([torch.nn.Parameter(torch.ones(2))]),
        torch.nn.ParameterDict({"weight": torch.nn.Parameter(torch.ones(2))}),
    ],
)
def test_nonexecutable_registry_container_boundary_is_rejected_before_mutation(provider, container):
    model = torch.nn.Module()
    model.container = container

    with pytest.raises((ValueError, TypeError), match="container|executable|ModuleList|ModuleDict|Parameter"):
        prepare(model, provider, ft_config=provider.OffloadConfig(include_patterns=["container"]))

    assert model.container is container
    assert provider.calls == []


def test_container_with_a_real_forward_can_be_prepared(provider):
    class ExecutableLayers(torch.nn.ModuleList):
        def forward(self, value):
            for layer in self:
                value = layer(value)
            return value

    model = torch.nn.Module()
    model.layers = ExecutableLayers([torch.nn.Linear(2, 2)])
    original = model.layers

    result = prepare(model, provider, ft_config=provider.OffloadConfig(include_patterns=["layers"]))

    assert result.unit_paths == ("layers",)
    assert model.layers.compute is original


def test_preexisting_shell_is_rejected_even_with_a_fresh_manager(provider):
    model = Model()
    shell = FlexTensorUnit(model.layers[0])
    model.layers[0] = shell

    with pytest.raises((ValueError, RuntimeError), match="shell|already|prepared|twice"):
        prepare(model, provider)

    assert model.layers[0] is shell
    assert not isinstance(model.layers[1], FlexTensorUnit)
    assert provider.calls == []


@pytest.mark.parametrize("name", ["literal*", "literal?", " padded", "padded "])
def test_unrepresentable_literal_module_names_fail_before_mutation(provider, name):
    model = torch.nn.Module()
    original = torch.nn.Linear(2, 2)
    model.add_module(name, original)

    with pytest.raises((ValueError, RuntimeError), match="literal|represent|wildcard|whitespace|name|path"):
        prepare(model, provider, ft_config=provider.OffloadConfig(include_patterns=["class:Linear"]))

    assert model.get_submodule(name) is original
    assert provider.calls == []


@pytest.mark.parametrize("name", ["literal*", "literal?", " padded", "padded "])
def test_unrepresentable_kept_parameter_names_fail_before_mutation(provider, name):
    model = Model()
    model.register_parameter(name, torch.nn.Parameter(torch.ones(2)))
    original = tuple(model.layers)

    with pytest.raises((ValueError, RuntimeError), match="literal|represent|wildcard|whitespace|name|path"):
        prepare(model, provider)

    assert_unprepared(model, original)
    assert provider.calls == []


def test_literal_wildcard_offloaded_parameter_does_not_need_exact_encoding(provider):
    model = Model()
    original = model.layers[0]
    original.register_parameter("literal*", torch.nn.Parameter(torch.ones(2)))

    result = prepare(model, provider)

    assert result.unit_paths == ("layers.0", "layers.1")
    assert model.layers[0].compute is original
    assert not any("literal*" in pattern for pattern in result.effective_config.exclude_patterns)


@pytest.mark.parametrize("alias_kind", ["module", "parameter", "parameter_storage", "buffer", "buffer_storage"])
def test_cross_boundary_aliases_fail_before_mutation(provider, alias_kind):
    model = Model()
    if alias_kind == "module":
        model.layers[1] = model.layers[0]
    elif alias_kind == "parameter":
        model.layers[1].projection.weight = model.layers[0].projection.weight
    elif alias_kind == "parameter_storage":
        model.layers[1].projection.weight = torch.nn.Parameter(model.layers[0].projection.weight.detach())
    elif alias_kind == "buffer":
        model.layers[1].offset = model.layers[0].offset
    else:
        model.layers[1].offset = model.layers[0].offset.view_as(model.layers[0].offset)
    original = tuple(model.layers)

    with pytest.raises((ValueError, RuntimeError), match="alias|shar|boundar"):
        prepare(model, provider)

    assert_unprepared(model, original)
    assert provider.calls == []


def test_parameter_alias_within_one_compute_boundary_does_not_require_copy(provider):
    model = Model()
    original = model.layers[0]
    original.norm.bias = original.projection.bias
    shared = original.projection.bias

    result = prepare(model, provider)

    assert result.unit_paths == ("layers.0", "layers.1")
    assert model.layers[0].compute.projection.bias is shared
    assert model.layers[0].compute.norm.bias is shared


def test_importing_preparation_api_does_not_require_flextensor():
    source = """
import builtins
original_import = builtins.__import__
def without_provider(name, *args, **kwargs):
    if name == 'flextensor' or name.startswith('flextensor.'):
        raise AssertionError('FT must not be imported eagerly')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_provider
from aitune.torch.jit.flextensor import prepare_flextensor_units, PreparedFlexTensor
assert callable(prepare_flextensor_units)
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=Path(__file__).resolve().parents[4],
    )
    assert result.returncode == 0, result.stdout + result.stderr
