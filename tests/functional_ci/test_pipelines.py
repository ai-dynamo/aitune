# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Functional CI coverage for toy FLUX pipelines."""

from functools import partial
from pathlib import Path
from typing import cast

import pytest
import torch
from packaging.version import Version
from torch import nn

import aitune.torch as ait
from aitune.torch.backend import KernelSelectorBackend
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.patched_module import ModuleState, PatchedModule
from aitune.torch.jit.patcher import prepare_for_jit_tuning
from aitune.torch.module import Module
from aitune.torch.module.tuned_module import TunedModule
from tests.functional_ci.backends import BackendCase, with_backends
from tests.functional_ci.flux import ToyFluxPipeline
from tests.functional_ci.lifecycle import Tolerance, pinned_strategy, selected_backend

TORCH_2_8_TO_2_10 = Version(torch.__version__).release[:2] in ((2, 8), (2, 9), (2, 10))

PROMPT = "A futuristic cityscape with neon lights and flying cars"
COMPONENTS = ("transformer", "decoder", "text_encoder", "text_encoder_2")


def generate(pipeline: ToyFluxPipeline, prompt: str | list[str]) -> torch.Tensor:
    """Generate deterministic toy images."""
    return pipeline(prompt, generator=torch.Generator(device=pipeline.device).manual_seed(0))


def _components(pipeline: ToyFluxPipeline) -> dict[str, Module]:
    """Return the four wrapped pipeline components."""
    components = {
        "transformer": cast(Module, pipeline.transformer),
        "decoder": cast(Module, pipeline.vae.decoder),
        "text_encoder": cast(Module, pipeline.text_encoder),
        "text_encoder_2": cast(Module, pipeline.text_encoder_2),
    }
    assert all(isinstance(component, Module) for component in components.values())
    return components


# Strict: a fixed TensorRT fails CI with XPASS until the mark is removed.
@pytest.mark.matrix("models.pipelines", "modes.aot-inspect-wrap", "modes.aot-save-load")
@with_backends(
    "save_load",
    marks={
        "torch-tensorrt-aot": pytest.mark.xfail(
            strict=True,
            reason=("torch_tensorrt/dynamo/conversion/_TRTInterpreter.py, line 703, in run assert cuda_engine"),
            raises=AssertionError,  # expected one selected backend, found 0",
        )
    },
)
def test_pipeline_lifecycle(case: BackendCase, device: torch.device, tmp_path: Path):
    """Inspect, tune, save, and restore the pipeline with each save/load backend."""
    backend = case.factory()
    pipeline = ToyFluxPipeline(device)
    expected = generate(pipeline, PROMPT)
    run = partial(generate, pipeline)

    inspected = ait.inspect(
        pipeline,
        [PROMPT],
        inference_function=run,
        number_of_iterations=1,
        warmup_iterations=1,
    )
    modules = inspected.get_modules()
    assert sorted(module.name for module in modules) == sorted(COMPONENTS)

    ait.wrap(pipeline, modules, strategy=pinned_strategy(backend))
    ait.tune(run, [PROMPT], batch_sizes=[1], device=device)

    for module in _components(pipeline).values():
        assert selected_backend(module, type(backend)).key() == backend.key()
        assert len(module.graph_specs) == 1
    case.tolerance.assert_close(generate(pipeline, PROMPT), expected)

    checkpoint = tmp_path / "pipeline.ait"
    ait.save(pipeline, checkpoint)
    for module in _components(pipeline).values():
        module.deactivate()

    restored = ait.load(ToyFluxPipeline(device, seed=1), checkpoint)
    assert isinstance(restored, ToyFluxPipeline)
    assert restored is not pipeline
    for module in _components(restored).values():
        selected_backend(module, type(backend))
    case.tolerance.assert_close(generate(restored, PROMPT), expected)


@pytest.mark.matrix("models.pipelines", "modes.jit-tune_eager")
@with_backends(
    "jit",
    marks={
        # Torch-TRT 2.12+ falls back to the original graph: the test passes, but the CLIP head runs in PyTorch.
        "torch-tensorrt-jit": pytest.mark.xfail(
            TORCH_2_8_TO_2_10,
            strict=True,
            reason=(
                "Torch-TRT < 2.12 rejects the int64 constant-folded CLIP position ids in `aten.embedding`, "
                "then its fallback fails with `AttributeError: _frozen_param0`, so the CLIP head is not tuned"
            ),
            raises=AssertionError,
        )
    },
)
def test_pipeline_jit(case: BackendCase, device: torch.device):
    """Tune the pipeline's four JIT heads from two prompt batch sizes."""
    expected = generate(ToyFluxPipeline(device), PROMPT)
    backend = case.factory()
    jit_config.mode = JITMode.TUNE_EAGER
    jit_config.device = device
    jit_config.min_samples = 2
    jit_config.strategy = ait.OneBackendStrategy(backend).enable_performance_validation(False)

    with prepare_for_jit_tuning():
        pipeline = ToyFluxPipeline(device)

    generate(pipeline, [PROMPT])
    generate(pipeline, [PROMPT] * 2)

    heads = PatchedModule.heads
    assert {head._name.split(" ")[0] for head in heads} == {
        "ToyClipTextEncoder",
        "ToyT5Encoder",
        "ToyFluxTransformer",
        "ToyDecoder",
    }
    assert all(head._state == ModuleState.TUNED for head in heads)
    assert all(
        type(selected) is type(backend)
        for head in heads
        for selected in cast(TunedModule, head._wrapper).backends.values()
    )
    case.tolerance.assert_close(generate(pipeline, PROMPT), expected)


@pytest.mark.matrix("models.pipelines/kernel-selector")
def test_pipeline_kernel_selector(device: torch.device):
    """Tune the Flux transformer with kernel selection and compare generated images."""
    pipeline = ToyFluxPipeline(device, dtype=torch.bfloat16)
    expected = generate(pipeline, PROMPT)
    transformer = ait.Module(
        cast(nn.Module, pipeline.transformer),
        name="transformer",
        strategy=pinned_strategy(KernelSelectorBackend()),
    )
    pipeline.transformer = transformer

    ait.tune(partial(generate, pipeline), [PROMPT], batch_sizes=[1], device=device)

    selector = cast(KernelSelectorBackend, selected_backend(transformer, KernelSelectorBackend))
    assert selector._runtime is not None and selector._runtime.plan.providers
    Tolerance(rtol=2e-2, atol=2e-2).assert_close(generate(pipeline, PROMPT), expected)
