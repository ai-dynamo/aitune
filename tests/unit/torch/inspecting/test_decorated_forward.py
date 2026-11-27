# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from copy import deepcopy
from functools import partial

import pytest
import torch
import wrapt
from packaging.version import Version

import aitune.torch as ait
from aitune.torch.backend import (
    Backend,
    TorchAOBackend,
    TorchAOBackendConfig,
    TorchInductorJitBackend,
    TorchInductorJitBackendConfig,
)
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.module.tuned_module import TunedModule
from aitune.torch.utils.torch_compile import DecoratedForwardBindings
from tests.toy_models.decorated_forward_models import ToyDecoratedSpeechModel
from tests.utilities.helpers import make_recorded_graph_spec, make_sample_store


@pytest.mark.parametrize("inspect_first", [False, True], ids=["fresh", "inspected"])
def test_decorated_forward_model_can_be_copied(inspect_first):
    model = ToyDecoratedSpeechModel().eval()
    sample = torch.ones(2, 8)
    if inspect_first:
        ait.inspect(model, [sample], number_of_iterations=1, warmup_iterations=1)

    copied = deepcopy(model)
    with torch.no_grad():
        original_output = model(sample)
        torch.testing.assert_close(copied(sample), original_output)
        copied.encoder.weight.zero_()
        # A copied forward must read the copy's weights even after inspection.
        torch.testing.assert_close(copied(sample), torch.zeros_like(original_output))
        torch.testing.assert_close(model(sample), original_output)


@pytest.mark.parametrize("instance_forward", [False, True], ids=["class-forward", "instance-forward"])
@pytest.mark.parametrize("inference_fails", [False, True], ids=["success", "failure"])
@pytest.mark.parametrize("call_tree", [False, True], ids=["without-call-tree", "with-call-tree"])
def test_inspection_restores_decorated_forward(instance_forward, inference_fails, call_tree):
    model = ToyDecoratedSpeechModel().eval()
    sample = torch.ones(2, 8)
    original_forward = model.encoder.forward
    if instance_forward:
        # Exercise an explicit instance override of wrapt's class descriptor.
        vars(model.encoder)["forward"] = original_forward

    with torch.no_grad():
        expected = model(sample)

    def inference(value):
        output = model(value)
        if inference_fails:
            raise RuntimeError("Inference failed during inspection")
        return output

    def inspect_model():
        ait.inspect(
            model,
            [sample],
            inference_function=inference,
            number_of_iterations=1,
            warmup_iterations=1,
            call_tree=call_tree,
        )

    if inference_fails:
        with pytest.raises(RuntimeError, match="Inference failed during inspection"):
            inspect_model()
    else:
        inspect_model()

    assert ("forward" in vars(model.encoder)) == instance_forward
    assert "forward" not in vars(model)
    if instance_forward:
        assert model.encoder.forward is original_forward
    with torch.no_grad():
        torch.testing.assert_close(model(sample), expected)


@pytest.mark.parametrize("inspect_first", [False, True], ids=["fresh", "inspected"])
@pytest.mark.xfail(
    Version(torch.__version__) >= Version("2.13.0a0"),
    raises=pytest.RaisesExc(AssertionError, match="FuncDefaultsGuardAccessor: Not a function"),
    reason="Dynamo cannot guard defaults on wrapt class descriptors (aitune#260); AITune binds them for compilation",
    strict=True,
)
def test_dynamo_compiles_decorated_forward_without_aitune_backend(inspect_first):
    model = ToyDecoratedSpeechModel().eval()
    sample = torch.ones(2, 8)
    if inspect_first:
        ait.inspect(model, [sample], number_of_iterations=1, warmup_iterations=1)

    torch._dynamo.reset()
    try:
        with torch.no_grad():
            expected = model(sample)
            # The eager compiler backend still exercises Dynamo's guards without requiring GPU kernels.
            compiled = torch.compile(model, backend="eager", fullgraph=False, dynamic=False)
            torch.testing.assert_close(compiled(sample), expected)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("inspect_first", [False, True], ids=["fresh", "inspected"])
def test_compilation_with_bound_decorated_forwards(inspect_first):
    model = ToyDecoratedSpeechModel().eval()
    sample = torch.ones(2, 8)
    if inspect_first:
        ait.inspect(model, [sample], number_of_iterations=1, warmup_iterations=1)

    bindings = DecoratedForwardBindings()
    torch._dynamo.reset()
    try:
        with torch.no_grad():
            expected = model(sample)
            with bindings.compiling(model):
                compiled = torch.compile(model, backend="eager", fullgraph=False, dynamic=False)
                torch.testing.assert_close(compiled(sample), expected)
            # Guards must remain valid after warmup, including when compilation is requested again.
            with bindings.compiling(model):
                torch.testing.assert_close(compiled(sample), expected)
            torch.testing.assert_close(compiled(sample + 1), model(sample + 1))
            # Binding must retain the decorator and its default and explicit arguments.
            torch.testing.assert_close(model.encoder(sample, scale=3), model.encoder(sample) * 1.5)
    finally:
        bindings.restore()
        torch._dynamo.reset()

    assert "forward" not in vars(model.encoder)
    with torch.no_grad():
        torch.testing.assert_close(deepcopy(model)(sample), expected)


def test_failed_compilation_restores_decorated_forward():
    model = ToyDecoratedSpeechModel().eval()
    bindings = DecoratedForwardBindings()
    with pytest.raises(RuntimeError, match="Compilation failed"):
        with bindings.compiling(model):
            raise RuntimeError("Compilation failed")

    assert "forward" not in vars(model.encoder)
    sample = torch.ones(2, 8)
    with torch.no_grad():
        torch.testing.assert_close(deepcopy(model)(sample), model(sample))


def test_bound_forward_preserves_runtime_decorator_checks():
    checks_enabled = [False]

    @wrapt.decorator(enabled=lambda: checks_enabled[0])
    def checked(wrapped, instance, args, kwargs):
        raise ValueError("Runtime check rejected the input")

    class CheckedModule(torch.nn.Module):
        # wrapt exposes a descriptor, although it remains callable as nn.Module.forward.
        @checked
        def forward(self, value):  # pyrefly: ignore[bad-override]
            return value + 1

    model = CheckedModule()
    bindings = DecoratedForwardBindings()
    try:
        bindings.bind(model)
        torch.testing.assert_close(model(torch.ones(1)), torch.full((1,), 2.0))
        checks_enabled[0] = True
        with pytest.raises(ValueError, match="Runtime check rejected the input"):
            model(torch.ones(1))
    finally:
        bindings.restore()


def test_decorated_forward_backend_checkpoint_reactivation(monkeypatch, tmp_path):
    # Use real Dynamo guards on CPU; GPU functional tests cover the compiled kernels.
    monkeypatch.setattr(torch, "compile", partial(torch.compile, backend="eager"))
    model = ToyDecoratedSpeechModel().eval()
    sample = torch.ones(2, 8)
    samples = make_sample_store([((sample,), {})], tmp_path)
    graph_spec = make_recorded_graph_spec(model, ((sample,), {}))
    backend = TorchInductorJitBackend(TorchInductorJitBackendConfig(dynamic=False))
    restored = None
    torch._dynamo.reset()
    try:
        with torch.no_grad():
            expected = model(sample)
        backend.build(model, graph_spec, samples, device=torch.device("cpu"), cache_dir=tmp_path)
        torch.testing.assert_close(backend.infer(sample), expected)
        state = backend.to_dict()
        backend.deactivate()
        torch.testing.assert_close(deepcopy(model)(sample), expected)

        fresh = ToyDecoratedSpeechModel(seed=1).eval()
        restored = TorchInductorJitBackend.from_dict(fresh, state)
        restored.activate()
        torch.testing.assert_close(restored.infer(sample), expected)
        restored.deactivate()
        restored.activate()
        torch.testing.assert_close(restored.infer(sample + 1), model(sample + 1))
    finally:
        if backend.is_active:
            backend.deactivate()
        if restored is not None and restored.is_active:
            restored.deactivate()
        torch._dynamo.reset()


def test_decorated_forward_checkpoint_activation_retries_after_warmup_failure(monkeypatch, tmp_path):
    compile_eager = partial(torch.compile, backend="eager")
    monkeypatch.setattr(torch, "compile", compile_eager)
    model = ToyDecoratedSpeechModel().eval()
    sample = torch.ones(2, 8)
    samples = make_sample_store([((sample,), {})], tmp_path)
    graph_spec = make_recorded_graph_spec(model, ((sample,), {}))
    backend = TorchInductorJitBackend(TorchInductorJitBackendConfig(dynamic=False))
    restored = None
    try:
        with torch.no_grad():
            expected = model(sample)
        backend.build(model, graph_spec, samples, device=torch.device("cpu"), cache_dir=tmp_path)
        state = backend.to_dict()
        backend.deactivate()
        restored = TorchInductorJitBackend.from_dict(ToyDecoratedSpeechModel(seed=1).eval(), state)

        def failing_warmup(*args, **kwargs):
            raise torch.OutOfMemoryError("Warmup allocation failed")

        def compile_with_failed_warmup(*args, **kwargs):
            # Subsequent compilations can succeed, but this failed artifact remains unusable.
            monkeypatch.setattr(torch, "compile", compile_eager)
            return failing_warmup

        monkeypatch.setattr(torch, "compile", compile_with_failed_warmup)
        with pytest.raises(torch.OutOfMemoryError, match="Warmup allocation failed"):
            restored.activate()
        assert not restored.is_active

        restored.activate()
        torch.testing.assert_close(restored.infer(sample), expected)
    finally:
        if backend.is_active:
            backend.deactivate()
        if restored is not None and restored.is_active:
            restored.deactivate()
        torch._dynamo.reset()


def test_decorated_forward_mixed_backend_checkpoint_activation(monkeypatch, tmp_path):
    monkeypatch.setattr(torch, "compile", partial(torch.compile, backend="eager"))
    model = ToyDecoratedSpeechModel().eval()
    inputs = (torch.ones(1, 8), torch.ones(2, 8))
    backends = [
        TorchInductorJitBackend(TorchInductorJitBackendConfig(dynamic=False)),
        TorchAOBackend(TorchAOBackendConfig(quantization="int8wo", mode="default", dynamic=False)),
    ]
    restored = None
    try:
        with torch.no_grad():
            expected = [model(sample) for sample in inputs]
        graph_specs = [make_recorded_graph_spec(model, ((sample,), {})) for sample in inputs]
        graph_backends: OrderedDict[SampleMetadata, Backend] = OrderedDict()
        for index, (backend, sample, graph_spec) in enumerate(zip(backends, inputs, graph_specs, strict=True)):
            cache_dir = tmp_path / str(index)
            cache_dir.mkdir()
            samples = make_sample_store([((sample,), {})], cache_dir)
            # Functional linear keeps outputs exact while exercising TorchAO's real model-copy path.
            backend.build(model, graph_spec, samples, device=torch.device("cpu"), cache_dir=cache_dir)
            graph_backends[graph_spec.input_spec] = backend
            backend.deactivate()

        tuned = TunedModule(graph_backends, "decorated", graph_specs[0].forward_signature)
        restored = TunedModule.from_dict(ToyDecoratedSpeechModel(seed=1).eval(), tuned.to_dict())
        restored.activate()
        for sample, reference in zip(inputs, expected, strict=True):
            torch.testing.assert_close(restored(sample), reference)
    finally:
        for backend in backends:
            if backend.is_active:
                backend.deactivate()
        if restored is not None:
            for backend in restored.backends.values():
                if backend.is_active:
                    backend.deactivate()
        torch._dynamo.reset()
