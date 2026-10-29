# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test device reporting for modules containing tuned submodules."""

from unittest.mock import Mock

import pytest
import torch

from aitune.torch.integrations.device import get_tuned_execution_device, patch_device_property
from aitune.torch.module.wrapper_module import Module, ModuleState
from aitune.torch.tune_strategy.tune_strategy import DummyTuneStrategy

EXECUTION_DEVICE = torch.device("cuda:1")


class Parent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(2, 2))
        self.decoder = torch.nn.Linear(2, 2)


def _wrap(parent: torch.nn.Module, name: str, tuned: bool = True) -> Module:
    wrapper = Module(getattr(parent, name), name, strategy=DummyTuneStrategy())
    setattr(parent, name, wrapper)
    if tuned:
        wrapper._self_state = ModuleState.TUNED
        wrapper._self_wrapper = Mock(device=EXECUTION_DEVICE)
    return wrapper


def test_get_tuned_execution_device_returns_device_of_tuned_owner():
    parent = Parent()
    _wrap(parent, "encoder").to("meta")

    assert get_tuned_execution_device(parent) == EXECUTION_DEVICE


def test_get_tuned_execution_device_ignores_untuned_owner():
    parent = Parent()
    _wrap(parent, "encoder", tuned=False)

    assert get_tuned_execution_device(parent) is None


def test_get_tuned_execution_device_ignores_tuned_module_not_owning_first_parameter():
    parent = Parent()
    _wrap(parent, "decoder").to("meta")

    assert get_tuned_execution_device(parent) is None


def test_get_tuned_execution_device_uses_buffers_without_parameters():
    parent = torch.nn.Module()
    parent.norm = torch.nn.BatchNorm1d(2, affine=False)
    _wrap(parent, "norm").to("meta")

    assert get_tuned_execution_device(parent) == EXECUTION_DEVICE


def _patched_model_class() -> type[Parent]:
    class Model(Parent):
        @property
        def device(self) -> torch.device:
            return next(self.parameters()).device

    patch_device_property(Model)
    return Model


def test_patch_device_property_reports_execution_device_of_offloaded_tuned_module():
    parent = _patched_model_class()()
    wrapper = _wrap(parent, "encoder")

    assert parent.device == torch.device("cpu")

    wrapper.to("meta")
    assert parent.device == EXECUTION_DEVICE


def test_patch_device_property_skips_class_without_device_property():
    class Model(Parent):
        device = torch.device("cpu")

    patch_device_property(Model)
    patch_device_property(Parent)

    assert Model.__dict__["device"] == torch.device("cpu")
    assert "device" not in Parent.__dict__


def test_patch_device_property_keeps_meta_without_tuned_owner():
    parent = _patched_model_class()().to("meta")

    assert parent.device == torch.device("meta")


def test_diffusers_model_mixin_device_is_patched():
    diffusers = pytest.importorskip("diffusers")

    class Model(diffusers.ModelMixin, Parent):
        pass

    parent = Model()
    _wrap(parent, "encoder").to("meta")

    assert parent.device == EXECUTION_DEVICE


def test_transformers_module_utils_mixin_device_is_patched():
    modeling_utils = pytest.importorskip("transformers.modeling_utils")

    class Model(modeling_utils.ModuleUtilsMixin, Parent):
        pass

    parent = Model()
    _wrap(parent, "encoder").to("meta")

    assert parent.device == EXECUTION_DEVICE
