---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: FlexTensor Integration
---

FlexTensor owns weight placement while AITune tunes the original computation inside each offload boundary. Install
[FlexTensor 0.5.0](https://pypi.org/project/flextensor/0.5.0/) separately; it remains optional for other AITune paths.
The adapter changes neither FlexTensor nor ordinary AITune compilation, validation, profiling or configuration.

## Residency boundaries

Wrap each selected module in `FlexTensorUnit` **before** FlexTensor prepares the model. Select these outer shells with
its include patterns, not their `compute` descendants. The original module remains `shell.compute`, with its parameters,
hooks and outputs; there is no independent resident weight copy.

FlexTensor's outer forward enters residency, calls the compute child, and exits only after that child returns. AITune
registers the child, so its normal warmups, correctness checks and performance measurements can repeat computation
without replaying provider transfer operations. Those transfers remain outside the inner Inductor graph.

An already prepared, unwrapped FlexTensor unit is unsafe for isolated tuning: its transfer operations can resubmit a
rolling slot before a future consumer releases it. The adapter rejects that preparation. It does not clear provider
state, join transfers mid-forward, or disable AITune checks to hide the issue.

## External compilation and explicit replan

This Torch 2.10 example assumes an uncompiled CPU `model` with `layers` and representative CUDA input `x`. Adapt the
selection, representative shapes and memory policy to the model. Leave headroom for inputs, activations and compilation.

```python
import torch
import flextensor as ft

from aitune.torch import jit_config
from aitune.torch.backend import TorchInductorJitBackend, TorchInductorJitBackendConfig
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.flextensor import FlexTensorUnit, register_flextensor_units, replan_flextensor
from aitune.torch.jit.tune import deferred
from aitune.torch.tune_strategy import OneBackendStrategy

model.layers = torch.nn.ModuleList(FlexTensorUnit(layer) for layer in model.layers)
jit_config.mode = JITMode.TUNE_DEFERRED
jit_config.device = "cuda"
jit_config.min_samples = 1
jit_config.batch_axis_required = False
jit_config.strategy = OneBackendStrategy(
    TorchInductorJitBackend(TorchInductorJitBackendConfig(fullgraph=True, mode="default"))
)

# Finite process-wide compiler settings belong to the application, not the adapter.
torch.compiler.config.recompile_limit = max(torch.compiler.config.recompile_limit, 128)
torch.compiler.config.accumulated_recompile_limit = max(torch.compiler.config.accumulated_recompile_limit, 512)

model = ft.offload(
    model.eval(),
    ft.OffloadConfig(
        include_patterns=["layers.*"],
        transfer_mode="allocation_block_transfer",
        profiling_iters=2,
        external_compile=True,
    ),
    name="aitune-example",
)
manager = ft.get_offload_manager("aitune-example")

def run_forward():
    return model(x)  # Exactly one complete model traversal.

with torch.no_grad():
    for _ in range(manager.iters_before_inference):
        run_forward()
    registration = register_flextensor_units([manager])
    run_forward()  # Record every representative shape/stage before deferred tuning.
    deferred()
    for _ in range(8):
        run_forward()
        if registration.all_tuned:
            break
    else:
        raise RuntimeError(f"Tuning did not complete: {registration.state_counts}")

    run_forward()  # Warm all selected serving routes before measuring their plan.
    replan_flextensor(manager, registration, run_forward)
    result = run_forward()
```

A video pipeline can traverse its transformer many times per generation. Record both stages before enabling deferred
tuning. The replan callback must execute **one** complete manager-model forward, not a generation or denoising loop.
The helper runs exactly the positive number of forwards returned by the public provider API and requires status
`state == "done"`, `applied is True`, and preserved live identities and dispatch.

## Retained view-profile strategy

For the separate retained-profile workflow, use `profile_mode="view"`, `external_compile=False`, and the identity
`compile_fn=lambda module: module` in `ft.offload` or `ft.offload_from_profile`. Finish provider startup, register the
final shells, record, tune and warm normally, but **do not call `replan_flextensor`**. Requesting a replan in this mode
raises an actionable error.

The identity callback activates the public compiled-profile path without compiling the shell. AITune later compiles
only its compute child. This retains the pre-AITune measured strategy, not compiled timings. It does not establish
compatibility with arbitrary `compile_fn` transformations. Validate outputs, actual residency and serving performance;
retention and completed replanning are distinct finalization modes.

## Registration and lifetime

`register_flextensor_units(managers, *, modules=None)` resolves each manager's final live public `offload_units()` after
it reaches inference. Omit `modules` for all shells or pass a subset of those public shells. The returned
`JITRegistration` view exposes the selected original **compute children**, reports and state counts. Register the complete
desired batch across managers once in a fresh worker.

Keep manager/model identities, shells, children and prepared forwards fixed afterward. Replanning validates these
associations and the installed AITune router/tuned dispatcher. The adapter installs no background coordinator hooks and
offers no reset, rollback, re-registration or recovery flow. Provider/callback exceptions propagate; stop that worker.

Ordinary AITune correctness checks, synchronous warmups, graph-break detection, autocast configuration, profiling,
dry-run, skip and native fallback stay on their existing paths. The enabled performance gate may select its measured
eager baseline, a real tuned selection rather than a compile failure. Native eager fallback, skips and dry-run
simulations do not themselves request a replan.

`all_tuned` does not prove acceleration. Verify selected backends, per-graph telemetry, outputs and measured performance.
`PerformanceValidationMode.DIAGNOSTIC` performs normal eager/candidate profiling without rejecting a slower candidate;
it is useful for coverage tests, but must not be described as disabled profiling or proof of a speedup.

## Compatibility and memory

The initial qualified backends/strategies are the existing
[external-placement candidates](jit_tuning.md#externally-owned-placement): exact eager/Inductor JIT classes and supported
Inductor modes/options. This adapter does not qualify other backends or CUDA-graph serving. Provider transfer operations
remain outside the pure-compute compiled graph.

The shell adds `.compute` to parameter/state-dict paths. It is not attribute/type-transparent and does not remap old
checkpoint keys. Adapt models that inspect block attributes before wrapping. Cross-tree parameter/storage aliases remain
outside the placement contract. Compute children must only run while their owner provides residency; standalone
offloaded-child inference and checkpoint deployment are not guaranteed.

Separate the strategy fraction, reported managed weight residency and whole-process GPU memory. Reserve known
preexisting residents when choosing the strategy budget. `get_gpu_memory_usage()` counts transfer blocks and all
unmapped tensors, not just `resident_units()`. Inputs, activations, workspace, allocator reserves and transient peaks need
separate measurement. A smaller fraction alone does not prove a strict total limit was met.

## Verification

Run each variant in a fresh interpreter:

```bash
python tests/functional/pytorch/jit/021_jit_flextensor.py --mode=eager
python tests/functional/pytorch/jit/021_jit_flextensor.py --mode=deferred
python tests/functional/pytorch/jit/021_jit_flextensor.py --mode=deferred --performance=enabled
python tests/functional/pytorch/jit/021_jit_flextensor.py --mode=deferred --finalization=profile
python tests/functional/pytorch/jit/022_jit_flextensor_fallback.py
```

Positive tests require all ten Inductor targets in diagnostic mode, normal correctness, actual per-graph profiling,
known/held-out numerical parity, real rolling storage under a strict reported weight-residency cap, one outer boundary
per traversal, and preserved live dispatch. The negative test requires a genuine compiler failure to preserve ordinary
eager fallback and provider residency without requesting a replan.

For the real pretrained all-48-block workload and manual video review, see the
[LTX 2.3 mini example](../../examples/LTX23OutpaintAITuneFlexTensor/README.md).
