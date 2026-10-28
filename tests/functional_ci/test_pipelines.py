# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Functional CI coverage for toy FLUX pipelines."""

from functools import partial
from pathlib import Path
from typing import cast

import pytest
import torch
from packaging.specifiers import SpecifierSet
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

PROMPT = "A futuristic cityscape with neon lights and flying cars"
COMPONENTS = ("transformer", "decoder", "text_encoder", "text_encoder_2")

# NGC builds are pre-releases (2.10.0a0+...), which sort below 2.10; `base_version` strips that.
TORCH_VERSION = Version(Version(torch.__version__).base_version)


class ComponentNotTunedError(Exception):
    """Raised when tuning leaves pipeline components without a backend."""


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
            raises=ComponentNotTunedError,
        ),
        # Torch-TRT cannot convert the CLIP text encoder: it rejects the int64 position ids passed to `aten.embedding`
        # and falls back to PyTorch for the whole encoder. On 2.10-2.12 that fallback crashes, so the encoder is not
        # tuned. On 2.8, 2.9, 2.13, and 2.14 the fallback works and the test passes, with CLIP running in PyTorch.
        "torch-tensorrt-jit": pytest.mark.xfail(
            TORCH_VERSION in SpecifierSet(">=2.10,<2.13"),
            strict=True,
            reason="Torch-TRT 2.10-2.12 crashes in its PyTorch fallback for the CLIP text encoder (`_frozen_param0`)",
            raises=ComponentNotTunedError,
        ),
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

    untuned = [name for name, module in _components(pipeline).items() if not getattr(module.module, "backends", None)]
    if untuned:
        raise ComponentNotTunedError(f"not tuned: {', '.join(untuned)}")
    for module in _components(pipeline).values():
        assert selected_backend(module, type(backend)).key() == backend.key()
        assert len(module.graph_specs) == 1
    case.tolerance.assert_close(generate(pipeline, PROMPT), expected)

    checkpoint = tmp_path / "pipeline.ait"
    ait.save(cast(nn.Module, pipeline), checkpoint)
    for module in _components(pipeline).values():
        module.deactivate()

    restored = cast(ToyFluxPipeline, ait.load(cast(nn.Module, ToyFluxPipeline(device, seed=1)), checkpoint))
    assert restored is not pipeline
    for module in _components(restored).values():
        selected_backend(module, type(backend))
    case.tolerance.assert_close(generate(restored, PROMPT), expected)


@pytest.mark.matrix("models.pipelines", "modes.jit-tune_eager")
@with_backends(
    "jit",
    marks={
        # Same Torch-TRT crash as in `test_pipeline_lifecycle`; the comment there explains it.
        "torch-tensorrt-jit": pytest.mark.xfail(
            TORCH_VERSION in SpecifierSet(">=2.10,<2.13"),
            strict=True,
            reason="Torch-TRT 2.10-2.12 crashes in its PyTorch fallback for the CLIP text encoder (`_frozen_param0`)",
            raises=ComponentNotTunedError,
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
    untuned = [head._name.split(" ")[0] for head in heads if head._state != ModuleState.TUNED]
    if untuned:
        raise ComponentNotTunedError(f"not tuned: {', '.join(untuned)}")
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
