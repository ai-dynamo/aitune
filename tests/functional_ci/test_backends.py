# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Functional backend lifecycle cases."""

from collections.abc import Callable
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import aitune.torch as ait
from aitune.torch.backend import (
    Backend,
    BuildMode,
    TensorRTBackend,
    TensorRTBackendConfig,
    TorchEagerBackend,
    TorchInductorAotBackend,
    TorchInductorJitBackend,
    TorchInductorJitBackendConfig,
    TorchTensorRTAotBackend,
    TorchTensorRTAotBackendConfig,
    TorchTensorRTConfig,
    TorchTensorRTJitBackend,
    TorchTensorRTJitBackendConfig,
)
from aitune.torch.module import Module

from .fixtures.artifacts import run_artifact
from .fixtures.assertions import selected_backend
from .fixtures.models import make_mlp


def _inductor_jit_backend() -> Backend:
    """Build a fixed-shape, full-graph Inductor JIT backend."""
    return TorchInductorJitBackend(TorchInductorJitBackendConfig(fullgraph=True, dynamic=False))


def _torch_tensorrt_jit_backend() -> Backend:
    """Build a Torch-TensorRT JIT backend that fully converts the graph to fp32."""
    return TorchTensorRTJitBackend(
        TorchTensorRTJitBackendConfig(
            compile_config=TorchTensorRTConfig(enabled_precisions={torch.float32}, min_block_size=1, disable_tf32=True)
        )
    )


def _torch_tensorrt_aot_backend() -> Backend:
    """Build a Torch-TensorRT AOT backend that fully converts the graph to fp32."""
    return TorchTensorRTAotBackend(
        TorchTensorRTAotBackendConfig(
            compile_config=TorchTensorRTConfig(enabled_precisions={torch.float32}, min_block_size=1, disable_tf32=True)
        )
    )


def _tensorrt_backend() -> Backend:
    """Build a standalone TensorRT backend with its lifecycle isolated from TF32 and CUDA graphs."""
    return TensorRTBackend(TensorRTBackendConfig(enable_tf32=False, use_cuda_graphs=False))


@pytest.mark.backend_lifecycle
@pytest.mark.parametrize(
    ("backend_factory", "device_mode", "expected_build_mode", "deployable"),
    [
        pytest.param(
            TorchEagerBackend,
            "explicit",
            BuildMode.JUST_IN_TIME,
            False,
            id="torch-eager-fp32",
            marks=pytest.mark.functional_case(backend="TorchEagerBackend", variant="fp32-explicit"),
        ),
        pytest.param(
            _inductor_jit_backend,
            "inferred",
            BuildMode.JUST_IN_TIME,
            False,
            id="inductor-jit-fp32",
            marks=pytest.mark.functional_case(backend="TorchInductorJitBackend", variant="fp32-jit-inferred"),
        ),
        pytest.param(
            TorchInductorAotBackend,
            "explicit",
            BuildMode.AHEAD_OF_TIME,
            True,
            id="inductor-aot-fp32",
            marks=pytest.mark.functional_case(backend="TorchInductorAotBackend", variant="fp32-aot-explicit"),
        ),
        pytest.param(
            _torch_tensorrt_jit_backend,
            "inferred",
            BuildMode.JUST_IN_TIME,
            False,
            id="torch-tensorrt-jit-fp32",
            marks=pytest.mark.functional_case(backend="TorchTensorRTJitBackend", variant="fp32-jit-inferred"),
        ),
        pytest.param(
            _torch_tensorrt_aot_backend,
            "explicit",
            BuildMode.AHEAD_OF_TIME,
            False,
            id="torch-tensorrt-aot-fp32",
            marks=pytest.mark.functional_case(backend="TorchTensorRTAotBackend", variant="fp32-aot-explicit"),
        ),
        pytest.param(
            _tensorrt_backend,
            "explicit",
            BuildMode.AHEAD_OF_TIME,
            True,
            id="tensorrt-fp32",
            marks=pytest.mark.functional_case(backend="TensorRTBackend", variant="fp32-explicit"),
        ),
    ],
)
def test_backend_lifecycle(
    functional_device: torch.device,
    tmp_path,
    backend_factory: Callable[[], Backend],
    device_mode: str,
    expected_build_mode: BuildMode,
    deployable: bool,
):
    """Tune, infer, save, restore, and natively run a backend where it is deployable."""
    requested_backend = backend_factory()
    strategy = ait.OneBackendStrategy(requested_backend).enable_performance_validation(False)
    model = make_mlp(device=functional_device)
    generator = torch.Generator(device=functional_device).manual_seed(1)
    value = torch.randn(2, 128, device=functional_device, generator=generator)
    with torch.no_grad():
        expected = model(value)

    inspected = ait.inspect(model, [value], number_of_iterations=1, warmup_iterations=1)
    module = ait.wrap(model, inspected.get_modules(), strategy=strategy)
    tune_device = functional_device if device_mode == "explicit" else None
    ait.tune(module, list(value.unbind(0)), batch_sizes=[2], device=tune_device)

    backend = selected_backend(module)
    assert type(backend) is type(requested_backend)
    assert backend.key() == requested_backend.key()
    assert backend.build_mode is expected_build_mode
    assert backend.device == value.device
    graph_spec = module.graph_specs[0]
    assert len(module.graph_specs) == 1
    assert graph_spec.input_spec.tensor_data[0][1].shape == [2, 128]
    assert graph_spec.output_spec.tensor_data[0][1].shape == [2, 64]

    _assert_real_execution(backend, module, value, expected)

    checkpoint = tmp_path / "backend.ait"
    ait.save(module, checkpoint)
    module.deactivate()
    restored = ait.load(make_mlp(device=functional_device), checkpoint)
    assert restored is not module
    restored_backend = selected_backend(restored)
    assert restored_backend is not backend
    assert restored_backend.build_mode is expected_build_mode
    torch.testing.assert_close(restored(value), expected, rtol=1e-4, atol=1e-5)

    if deployable:
        _assert_native_artifact(restored, tmp_path, value, expected)


def _assert_real_execution(backend: Backend, module: Module, value: torch.Tensor, expected: torch.Tensor) -> None:
    """Confirm the real runtime ran: a compiled region for JIT, an executed engine or runner for AOT."""
    if isinstance(backend, TorchInductorJitBackend):
        assert backend._compiled_module is not None
        torch.testing.assert_close(module(value), expected, rtol=1e-4, atol=1e-5)
        return
    if isinstance(backend, TorchInductorAotBackend):
        runner = backend._runner
        assert runner is not None
        calls = 0

        def observed_runner(*args, **kwargs):
            nonlocal calls
            calls += 1
            return runner(*args, **kwargs)

        backend._runner = observed_runner
        try:
            torch.testing.assert_close(module(value), expected, rtol=1e-4, atol=1e-5)
        finally:
            backend._runner = runner
        assert calls == 1
        return
    if isinstance(backend, (TorchTensorRTJitBackend, TorchTensorRTAotBackend)):
        with _observed_torch_tensorrt_execution() as counter:
            torch.testing.assert_close(module(value), expected, rtol=1e-4, atol=1e-5)
        assert counter.calls == 1
        return
    if isinstance(backend, TensorRTBackend):
        assert backend._config.enable_tf32 is False
        assert backend._config.use_cuda_graphs is False
        original = backend._execute_engine
        calls = 0

        def observed_execute():
            nonlocal calls
            calls += 1
            return original()

        backend._execute_engine = observed_execute
        try:
            torch.testing.assert_close(module(value), expected, rtol=1e-4, atol=1e-5)
        finally:
            backend._execute_engine = original
        assert calls == 1
        return
    torch.testing.assert_close(module(value), expected, rtol=1e-4, atol=1e-5)


@contextmanager
def _observed_torch_tensorrt_execution():
    """Count real Torch-TensorRT engine runs via the ``tensorrt::execute_engine`` operator."""
    counter = SimpleNamespace(calls=0)
    packet = torch.ops.tensorrt.execute_engine

    class _ObservedPacket:
        def __call__(self, *args, **kwargs):
            counter.calls += 1
            return packet(*args, **kwargs)

        def __getattr__(self, item):
            target = getattr(packet, item)
            if item != "default":
                return target

            def wrapper(*args, **kwargs):
                counter.calls += 1
                return target(*args, **kwargs)

            return wrapper

    torch.ops.tensorrt.execute_engine = _ObservedPacket()
    try:
        yield counter
    finally:
        torch.ops.tensorrt.execute_engine = packet


_NATIVE_RUNTIME_NAMES = {"pt2": "aotinductor", "tensorrt_plan": "tensorrt"}
_NATIVE_FILE_NAMES = {"pt2": "model.pt2", "tensorrt_plan": "model.plan"}


def _assert_native_artifact(module: Module, tmp_path, value: torch.Tensor, expected: torch.Tensor) -> None:
    """Export the deployment artifact to a new path and execute it through its native runtime."""
    artifact = module.artifact()
    model_format = artifact.model.format
    assert artifact.runtime.name == _NATIVE_RUNTIME_NAMES[model_format]

    exported_path = tmp_path / "exported" / _NATIVE_FILE_NAMES[model_format]
    assert artifact.model.export_files(exported_path) == exported_path

    outputs = run_artifact(artifact, exported_path, {artifact.input_names[0]: value})
    native_output = outputs[artifact.output_names[0]]
    assert native_output.shape == (2, 64)
    torch.testing.assert_close(native_output, expected, rtol=1e-4, atol=1e-5)
