# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the composite kernel optimizer backend."""

import json
from concurrent.futures import Future
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.nn.attention import SDPBackend

from aitune.torch.backend.backend import Backend, BackendState, BuildMode, ExecutionMode
from aitune.torch.backend.kernel_optimizer_backend import (
    KernelOptimizerBackend,
    KernelOptimizerBackendConfig,
)
from aitune.torch.backend.kernels.kernel_optimization_plan import KernelOptimizationPlan
from aitune.torch.backend.kernels.kernel_provider import (
    DiffusersAttentionBackend,
    DiffusersAttentionKernelProvider,
    FlashAttention4KernelProvider,
    KernelGenerationResult,
    KernelGenerator,
    KernelProvider,
    TorchSDPAKernelProvider,
)
from aitune.torch.backend.torch_inductor_jit_backend import TorchInductorJitBackend
from tests.utilities.helpers import make_sample_store


class _ReluProvider(KernelProvider):
    """Serializable provider that adds a constant after ReLU."""

    def __init__(self, increment: float = 0.0):
        super().__init__()
        self.increment = increment
        self.calls = 0

    @property
    def supported_function(self) -> str:
        return "relu"

    @property
    def name(self) -> str:
        return f"test-relu-{self.increment}"

    def _prepare(self, samples) -> bool:
        return True

    def _infer(self, value):
        self.calls += 1
        return torch.relu(value) + self.increment

    def _to_dict(self):
        return {"increment": self.increment}

    @classmethod
    def _from_dict(cls, state_dict):
        return cls(state_dict["increment"])


class _ReluGenerator(KernelGenerator):
    """Generator used only to verify configuration plumbing."""

    def __repr__(self) -> str:
        return "test-relu-generator"

    @property
    def name(self) -> str:
        return "test-relu-generator"

    def supports_functions(self) -> list[str]:
        return ["relu"]

    def prepare(self, function, samples) -> bool:
        return True

    def submit(self, function, samples) -> Future[KernelGenerationResult]:
        future = Future()
        provider = _ReluProvider()
        provider.prepare(samples)
        future.set_result(KernelGenerationResult(function, provider, repr(self)))
        return future


class _ReluModule(nn.Module):
    def forward(self, value):
        return F.relu(value)


class _PlanOptimizer:
    """Return a prepared plan without invoking the CUDA profiler."""

    def __init__(self, plan):
        self.plan = plan
        self.make_plan_calls = []

    def make_plan(self, function, data, *, module):
        self.make_plan_calls.append((function, data, module))
        return self.plan


class _RecordingJitDelegate(Backend):
    _build_mode = BuildMode.JUST_IN_TIME
    _execution_modes = frozenset({ExecutionMode.SINGLE_GPU, ExecutionMode.MULTI_GPU})

    def __init__(self):
        super().__init__()
        self.module = None
        self.sample = None
        self.cache_dir = None
        self.build_output = None
        self.activate_calls = 0
        self.deactivate_calls = 0
        self.deploy_calls = 0

    def key(self) -> str:
        return self.__class__.__name__

    def describe(self) -> str:
        return self.__class__.__name__

    def _build(self, module, graph_spec, samples, cache_dir):
        self.module = module
        self.sample = samples[0]
        self.cache_dir = cache_dir
        self.build_output = self.module(*self.sample[0], **self.sample[1])
        return self

    def _activate(self):
        self.activate_calls += 1
        if self.sample is not None:
            self.module(*self.sample[0], **self.sample[1])

    def _infer(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _deactivate(self):
        self.deactivate_calls += 1

    def _deploy(self):
        self.deploy_calls += 1
        self._activate()

    def to_dict(self):
        return {
            "type": self.__class__.__name__,
            "device": self._device,
        }

    @classmethod
    def from_dict(cls, module, state_dict):
        backend = cls()
        backend.module = module
        backend._device = state_dict["device"]
        backend.state = BackendState.CHECKPOINT_LOADED
        return backend


class _RecordingAotDelegate(_RecordingJitDelegate):
    _build_mode = BuildMode.AHEAD_OF_TIME
    _execution_modes = frozenset({ExecutionMode.SINGLE_GPU})

    @classmethod
    def from_dict(cls, module, state_dict):
        backend = cls()
        backend._device = state_dict["device"]
        backend.state = BackendState.CHECKPOINT_LOADED
        return backend


class _FailingDelegate(_RecordingJitDelegate):
    _build_mode = BuildMode.JUST_IN_TIME
    _execution_modes = frozenset({ExecutionMode.SINGLE_GPU})

    def _build(self, module, graph_spec, samples, cache_dir):
        self.module = module
        self.sample = samples[0]
        self.module(*self.sample[0], **self.sample[1])
        raise RuntimeError("delegate build failed")


def _prepared_plan(increment=10.0):
    provider = _ReluProvider(increment)
    provider.prepare([])
    return KernelOptimizationPlan((provider,)), provider


def _backend(delegate, optimizer):
    source = _ReluProvider()
    backend = KernelOptimizerBackend(
        config=KernelOptimizerBackendConfig(kernel_providers=source),
        delegate_backend=delegate,
    )
    backend._create_optimizer = Mock(return_value=optimizer)
    return backend


def _build(backend, tmp_path):
    module = _ReluModule()
    sample = ((torch.tensor([-1.0, 1.0]),), {})
    samples = make_sample_store([sample], tmp_path, "samples")
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    backend.build(module, Mock(), samples, torch.device("cpu"), build_dir)
    return module, sample, samples


def test_config_defaults_to_attention_providers():
    first_config = KernelOptimizerBackendConfig()
    second_config = KernelOptimizerBackendConfig()
    first_providers = first_config.kernel_providers
    second_providers = second_config.kernel_providers

    assert isinstance(first_providers, list)
    assert isinstance(second_providers, list)
    assert [type(provider) for provider in first_providers] == [
        TorchSDPAKernelProvider,
        TorchSDPAKernelProvider,
        DiffusersAttentionKernelProvider,
        FlashAttention4KernelProvider,
    ]
    cudnn_provider, flash_provider, flash_3_provider, _ = first_providers
    assert isinstance(cudnn_provider, TorchSDPAKernelProvider)
    assert cudnn_provider.backend is SDPBackend.CUDNN_ATTENTION
    assert isinstance(flash_provider, TorchSDPAKernelProvider)
    assert flash_provider.backend is SDPBackend.FLASH_ATTENTION
    assert isinstance(flash_3_provider, DiffusersAttentionKernelProvider)
    assert flash_3_provider.backend is DiffusersAttentionBackend.FLASH_3_HUB
    assert first_providers is not second_providers
    provider_pairs = zip(first_providers, second_providers, strict=True)
    assert all(first is not second for first, second in provider_pairs)


def test_config_requires_and_normalizes_explicit_kernel_sources():
    with pytest.raises(ValueError, match="At least one kernel provider or generator must be provided"):
        KernelOptimizerBackendConfig(kernel_providers=None)

    provider = _ReluProvider()
    provider_config = KernelOptimizerBackendConfig(kernel_providers=provider)
    generator = _ReluGenerator()
    generator_config = KernelOptimizerBackendConfig(kernel_providers=None, kernel_generators=generator)

    assert provider_config.kernel_providers == [provider]
    assert provider_config.kernel_generators == []
    assert generator_config.kernel_providers == []
    assert generator_config.kernel_generators == [generator]


def test_backend_defaults_to_default_config_and_torch_inductor_jit_delegate():
    first_backend = KernelOptimizerBackend()
    second_backend = KernelOptimizerBackend()

    assert isinstance(first_backend._config, KernelOptimizerBackendConfig)
    assert isinstance(first_backend._delegate_backend, TorchInductorJitBackend)
    assert first_backend._config is not second_backend._config
    assert first_backend._delegate_backend is not second_backend._delegate_backend


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("provider_min_time_share_percent", 101.0, "provider_min_time_share_percent"),
        ("generator_min_time_share_percent", -1.0, "generator_min_time_share_percent"),
        ("generation_timeout", -1.0, "generation_timeout"),
    ],
)
def test_config_rejects_invalid_values(field, value, match):
    kwargs = {"kernel_providers": _ReluProvider(), field: value}

    with pytest.raises((TypeError, ValueError), match=match):
        KernelOptimizerBackendConfig(**kwargs)


def test_config_key_and_description_include_kernel_sources():
    provider = _ReluProvider(1.0)
    config = KernelOptimizerBackendConfig(
        kernel_providers=provider,
        kernel_generators=_ReluGenerator(),
        provider_min_time_share_percent=5.0,
        generator_min_time_share_percent=20.0,
        generation_timeout=30.0,
    )

    initial_key = config.key()
    config_dict = config.to_dict()

    assert json.loads(json.dumps(config_dict)) == config_dict
    assert config_dict["kernel_providers"] == ["test-relu-1.0"]
    assert config_dict["kernel_generators"] == ["test-relu-generator"]
    assert "kernel_providers=['test-relu-1.0']" in config.describe()
    assert "kernel_generators=['test-relu-generator']" in config.describe()
    assert "generation_timeout=30.0" in config.describe()

    provider.prepare([])
    assert config.key() == initial_key


def test_config_deserialization_is_not_supported():
    with pytest.raises(NotImplementedError, match="one-way serialization only"):
        KernelOptimizerBackendConfig.from_dict({})


@pytest.mark.parametrize(
    ("delegate"),
    [
        _RecordingJitDelegate(),
        _RecordingAotDelegate(),
    ],
)
def test_backend_adopts_delegate_modes(delegate):
    plan, _ = _prepared_plan()
    backend = _backend(delegate, _PlanOptimizer(plan))

    assert backend._build_mode is delegate._build_mode
    assert backend._execution_modes == delegate._execution_modes


def test_create_optimizer_uses_backend_config(mocker):
    provider = _ReluProvider()
    generator = _ReluGenerator()
    config = KernelOptimizerBackendConfig(
        kernel_providers=provider,
        kernel_generators=generator,
        provider_min_time_share_percent=5.0,
        generator_min_time_share_percent=20.0,
        generation_timeout=30.0,
    )
    backend = KernelOptimizerBackend(config, _RecordingJitDelegate())
    optimizer_class = mocker.patch("aitune.torch.backend.kernel_optimizer_backend.KernelOptimizer")

    optimizer = backend._create_optimizer()

    assert optimizer is optimizer_class.return_value
    optimizer_class.assert_called_once_with(
        kernel_providers=[provider],
        kernel_generators=[generator],
        provider_min_time_share_percent=5.0,
        generator_min_time_share_percent=20.0,
        generation_timeout=30.0,
    )


def test_build_makes_plan_and_builds_delegate(tmp_path):
    plan, selected_provider = _prepared_plan()
    optimizer = _PlanOptimizer(plan)
    backend = _backend(_RecordingJitDelegate(), optimizer)

    original_module, _, samples = _build(backend, tmp_path)
    delegate = backend._delegate_backend

    assert backend.state is BackendState.ACTIVE
    assert backend._runtime.module is original_module
    backend._create_optimizer.assert_called_once_with()
    assert optimizer.make_plan_calls == [(original_module, samples, original_module)]
    assert isinstance(delegate, _RecordingJitDelegate)
    assert delegate.module is original_module
    assert delegate.cache_dir == tmp_path / "build" / "delegate_backend"
    synchronized_provider = backend._runtime.plan.providers[0]
    assert isinstance(synchronized_provider, _ReluProvider)
    assert synchronized_provider is not selected_provider
    assert selected_provider.calls == 0
    assert synchronized_provider.calls == 1
    assert backend._runtime.is_active
    assert backend._build_results == [
        {
            "detailed_build_info": {
                "local_optimization_plan": plan.to_dict(),
                "synchronized_plan": plan.to_dict(),
            }
        },
    ]


def test_synchronize_plan_restores_rank_zero_plan_on_nonzero_rank(mocker):
    local_plan, local_provider = _prepared_plan(increment=20.0)
    rank_zero_plan, _ = _prepared_plan(increment=10.0)
    backend = _backend(_RecordingJitDelegate(), _PlanOptimizer(local_plan))
    broadcast = mocker.patch(
        "aitune.torch.backend.kernel_optimizer_backend.coordinator.broadcast_from_rank0",
        return_value=rank_zero_plan.to_dict(),
    )

    synchronized_plan = backend._synchronize_plan(local_plan)

    assert synchronized_plan is not local_plan
    synchronized_provider = synchronized_plan.providers[0]
    assert isinstance(synchronized_provider, _ReluProvider)
    assert synchronized_provider.increment == 10.0
    assert synchronized_provider is not local_provider
    broadcast.assert_called_once_with(local_plan.to_dict())
    assert backend._build_results == [
        {
            "detailed_build_info": {
                "local_optimization_plan": local_plan.to_dict(),
                "synchronized_plan": rank_zero_plan.to_dict(),
            }
        },
    ]


def test_synchronize_plan_broadcasts_and_restores_rank_zero_state(mocker):
    plan, provider = _prepared_plan(increment=10.0)
    backend = _backend(_RecordingJitDelegate(), _PlanOptimizer(plan))
    broadcast = mocker.patch(
        "aitune.torch.backend.kernel_optimizer_backend.coordinator.broadcast_from_rank0",
        side_effect=lambda value: value,
    )

    synchronized_plan = backend._synchronize_plan(plan)

    assert synchronized_plan is not plan
    assert synchronized_plan.providers[0] is not provider
    assert synchronized_plan.to_dict() == plan.to_dict()
    broadcast.assert_called_once_with(plan.to_dict())


def test_jit_inference_uses_active_selected_provider(tmp_path):
    plan, _ = _prepared_plan()
    backend = _backend(_RecordingJitDelegate(), _PlanOptimizer(plan))
    _, sample, _ = _build(backend, tmp_path)
    synchronized_provider = backend._runtime.plan.providers[0]
    synchronized_provider.calls = 0
    original_relu = F.relu

    output = backend.infer(*sample[0], **sample[1])

    torch.testing.assert_close(output, torch.tensor([10.0, 11.0]))
    assert synchronized_provider.calls == 1
    assert F.relu is original_relu
    assert backend._runtime.is_active


def test_aot_backend_discards_runtime_after_delegate_build(tmp_path):
    plan, _ = _prepared_plan()
    backend = _backend(_RecordingAotDelegate(), _PlanOptimizer(plan))

    _, sample, _ = _build(backend, tmp_path)
    delegate = backend._delegate_backend
    output = backend.infer(*sample[0], **sample[1])

    assert isinstance(delegate, _RecordingAotDelegate)
    torch.testing.assert_close(delegate.build_output, torch.tensor([10.0, 11.0]))
    torch.testing.assert_close(output, torch.tensor([0.0, 1.0]))
    assert backend._runtime is None


def test_deactivation_and_reactivation_update_runtime(tmp_path):
    plan, _ = _prepared_plan()
    backend = _backend(_RecordingJitDelegate(), _PlanOptimizer(plan))
    _build(backend, tmp_path)
    delegate = backend._delegate_backend
    synchronized_provider = backend._runtime.plan.providers[0]
    synchronized_provider.calls = 0

    backend.deactivate()
    assert not backend._runtime.is_active

    backend.activate()

    assert synchronized_provider.calls == 1
    assert isinstance(delegate, _RecordingJitDelegate)
    assert delegate.deactivate_calls == 1
    assert backend._runtime.is_active


def test_failed_delegate_build_restores_function_and_hooks(tmp_path):
    plan, _ = _prepared_plan()
    backend = _backend(_FailingDelegate(), _PlanOptimizer(plan))
    original_relu = F.relu
    module = _ReluModule()
    sample = ((torch.tensor([-1.0, 1.0]),), {})
    samples = make_sample_store([sample], tmp_path, "samples")
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    with pytest.raises(RuntimeError, match="delegate build failed"):
        backend.build(module, Mock(), samples, torch.device("cpu"), build_dir)

    assert F.relu is original_relu
    assert len(module._forward_pre_hooks) == 0
    assert len(module._forward_hooks) == 0
    assert backend._runtime is None


def test_checkpoint_restores_inactive_jit_runtime_without_optimizer(tmp_path, monkeypatch):
    plan, _ = _prepared_plan()
    backend = _backend(_RecordingJitDelegate(), _PlanOptimizer(plan))
    _, sample, _ = _build(backend, tmp_path)
    state = backend.to_dict()

    class _UnexpectedOptimizer:
        def __init__(self, **kwargs):
            raise AssertionError("KernelOptimizer must not be created while loading")

    monkeypatch.setattr("aitune.torch.backend.kernel_optimizer_backend.KernelOptimizer", _UnexpectedOptimizer)
    restored_module = _ReluModule()
    restored = KernelOptimizerBackend.from_dict(restored_module, state)
    assert not restored._runtime.is_active

    restored.deploy(torch.device("cpu"))
    output = restored.infer(*sample[0], **sample[1])

    torch.testing.assert_close(output, torch.tensor([10.0, 11.0]))
    assert restored.state is BackendState.DEPLOYED
    assert restored._config is None
    assert restored._runtime.module is restored_module
    assert restored._delegate_backend.deploy_calls == 1
    assert restored._runtime.is_active


def test_checkpoint_jit_runtime_follows_activate_and_deactivate(tmp_path):
    plan, _ = _prepared_plan()
    backend = _backend(_RecordingJitDelegate(), _PlanOptimizer(plan))
    _, sample, _ = _build(backend, tmp_path)
    restored = KernelOptimizerBackend.from_dict(_ReluModule(), backend.to_dict())

    restored.activate()
    output = restored.infer(*sample[0], **sample[1])

    torch.testing.assert_close(output, torch.tensor([10.0, 11.0]))
    assert restored.state is BackendState.ACTIVE
    assert restored._runtime.is_active

    restored.deactivate()

    assert restored.state is BackendState.INACTIVE
    assert not restored._runtime.is_active


def test_checkpoint_restores_aot_delegate_without_module(tmp_path):
    plan, _ = _prepared_plan()
    backend = _backend(_RecordingAotDelegate(), _PlanOptimizer(plan))
    _build(backend, tmp_path)

    restored = KernelOptimizerBackend.from_dict(None, backend.to_dict())

    assert restored.state is BackendState.CHECKPOINT_LOADED
    assert restored.build_mode is BuildMode.AHEAD_OF_TIME
    assert restored._runtime is None
    assert KernelOptimizerBackend.STATE_KERNEL_PLAN not in backend.to_dict()


@pytest.mark.parametrize(
    ("state", "match"),
    [
        ({"type": "WrongBackend"}, "Invalid state_dict type"),
        ({"type": "KernelOptimizerBackend", "version": 2}, "Unsupported KernelOptimizerBackend state version"),
        (
            {"type": "KernelOptimizerBackend", "version": 1, "kernel_plan": {}},
            "requires a delegate backend",
        ),
    ],
)
def test_checkpoint_rejects_invalid_parent_state(state, match):
    with pytest.raises(ValueError, match=match):
        KernelOptimizerBackend.from_dict(None, state)


def test_checkpoint_rejects_unknown_delegate_type():
    state = {
        "type": "KernelOptimizerBackend",
        "version": 1,
        "kernel_plan": {"providers": []},
        "delegate_backend": {"type": "UnknownDelegate"},
    }

    with pytest.raises(ValueError, match="Unknown delegate backend type"):
        KernelOptimizerBackend.from_dict(None, state)


def test_checkpoint_requires_module_for_jit_delegate(tmp_path):
    plan, _ = _prepared_plan()
    backend = _backend(_RecordingJitDelegate(), _PlanOptimizer(plan))
    _build(backend, tmp_path)

    with pytest.raises(ValueError, match="Module is required to restore a just-in-time delegate"):
        KernelOptimizerBackend.from_dict(None, backend.to_dict())
