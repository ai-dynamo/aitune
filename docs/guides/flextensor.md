---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: FlexTensor Integration
---

FlexTensor owns weight placement while AITune tunes the original computation inside each offload boundary. Install
[FlexTensor 0.5.0](https://pypi.org/project/flextensor/0.5.0/) separately; it remains optional for other AITune paths.
The adapter changes neither FlexTensor nor ordinary AITune compilation, validation, profiling or configuration.

## Residency boundaries

Use `prepare_flextensor_units`, or manually insert `FlexTensorUnit` shells, **before** FlexTensor prepares the model.
FlexTensor selects these outer shells, not their `compute` descendants. The original module remains `shell.compute`,
with its parameters, hooks and outputs. Wrapping itself does not copy or move weights; FlexTensor still owns placement
and any provider-side snapshots.

FlexTensor's outer forward enters residency, calls the compute child, and exits only after that child returns. AITune
registers the child, so its normal warmups, correctness checks and performance measurements can repeat computation
without replaying provider transfer operations. Those transfers remain outside the inner Inductor graph.

An already prepared, unwrapped FlexTensor unit is unsafe for isolated tuning: its transfer operations can resubmit a
rolling slot before a future consumer releases it. Registration rejects units that are not `FlexTensorUnit` shells,
but structural checks cannot detect every incorrect manual preparation order. Adding shells after FT preparation
remains unsafe even if those checks pass. The preparation helper performs these operations in the correct order for
an unprepared model. It does not clear provider state, join transfers mid-forward, or disable AITune checks.

## Preparation helper

`prepare_flextensor_units(model_or_owner, *, ft_config, target=None, name="default", load_profile_from=None)` is an
optional setup convenience; the [manual path](#manual-setup) remains available. Supply a fresh, uncompiled model and
an explicit public `flextensor.OffloadConfig`. Selectors refer to the **original model**, before shells add `.compute`:
module names, class patterns, parameter includes and exclusions retain their meaning.

The helper plans the unit boundaries and parameter residency policy before changing the model. It inserts shells and
passes a validated shallow config copy to public FT setup. The caller's config is not mutated; unrelated settings,
including memory policy and custom strategy objects, are retained. No private FT selector or lifecycle API is used,
and the helper adds no private AITune core calls.

The returned `PreparedFlexTensor` exposes `model` (the exact FT-returned proxy), `manager`, `effective_config`, and
`unit_paths` (the planned boundaries in the original model). After startup, discover live units through the manager;
do not cache pre-startup module references. For a bare root, use `model = prepared.model`. FT's proxy follows its model
phase transitions; substituting the original model or a snapshot of `manager.model` is not equivalent.

Preparation runs no model forwards and does not register targets, trigger tuning, warm routes, or replan. It does not
move application-owned tensors or change JIT options, backends or compiler limits. The caller keeps these steps explicit.

## External compilation and explicit replan

This Torch 2.10 example assumes an uncompiled CPU `model` with `layers` and representative CUDA input `x`. Adapt the
selection, representative shapes and memory policy to the model. Leave headroom for inputs, activations and compilation.

```python
import torch
import flextensor as ft

from aitune.torch import jit_config
from aitune.torch.backend import TorchInductorJitBackend, TorchInductorJitBackendConfig
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.flextensor import prepare_flextensor_units, register_flextensor_units, replan_flextensor
from aitune.torch.jit.tune import deferred
from aitune.torch.tune_strategy import OneBackendStrategy

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

cfg = ft.OffloadConfig(
    include_patterns=["layers.*"],
    transfer_mode="allocation_block_transfer",
    profiling_iters=2,
    external_compile=True,
)
prepared = prepare_flextensor_units(
    model.eval(),
    ft_config=cfg,
    name="aitune-example",
)
model, manager = prepared.model, prepared.manager

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
The replan helper runs exactly the positive number of forwards returned by the public provider API and requires status
`state == "done"`, `applied is True`, and preserved live identities and dispatch.

### Preparing a pipeline attribute

For an existing pipeline whose transformer contains `transformer_blocks`, use this setup form instead of the bare-root
preparation above. The owner need not be an `nn.Module`:

```python
cfg = ft.OffloadConfig(
    include_patterns=["transformer_blocks.*"],
    external_compile=True,
)
prepared = prepare_flextensor_units(
    ltx_model,
    target="transformer",
    ft_config=cfg,
    name="ltx",
)
manager = prepared.manager
# ltx_model.transformer already contains prepared.model; no assignment is needed.
```

`target` is an explicit attribute path; dotted paths are supported. The helper uses ordinary attribute assignment and
does not guess parents, bypass setters, or replace other aliases. Run startup and inference through the installed proxy
or the normal pipeline, not an old transformer reference or isolated compute children. Application-specific placement
of direct parameters outside selected boundaries remains the caller's responsibility; see the LTX example.

## Manual setup

The existing manual setup remains supported. In a fresh worker, replace the helper's setup lines with the following
external-compilation setup, then follow the same startup, registration, tuning, warmup and replan sequence. Choose one
setup form; do not run the helper and manual setup on the same model.

```python
from aitune.torch.jit.flextensor import FlexTensorUnit

for index, layer in enumerate(model.layers):
    model.layers[index] = FlexTensorUnit(layer)
cfg = ft.OffloadConfig(include_patterns=["layers.*"], external_compile=True)
model = ft.offload(model.eval(), config=cfg, name="aitune-example")
manager = ft.get_offload_manager("aitune-example")
```

In this form, FT selectors describe the **wrapped tree**: select shell boundaries and adapt class/parameter/exclusion
patterns to that layout, including `.compute` where needed. The helper instead handles that translation from original
names. Neither form remaps checkpoint keys or makes wrappers attribute/type-transparent. Advanced public FT setup can
remain manual; that does not qualify arbitrary compiler callbacks or unsafe late wrapping for isolated AITune tuning.

## Retained view-profile strategy

For the separate retained-profile workflow, set `profile_mode="view"`, `external_compile=False` in `ft_config`.
The preparation helper selects an identity compile callback. With manual `ft.offload` or `ft.offload_from_profile`,
pass `compile_fn=lambda module: module` yourself. Finish provider startup, register the final shells, record, tune and
warm normally, but **do not call `replan_flextensor`**. Requesting a replan in this mode raises an actionable error.

The identity callback activates the public compiled-profile path without compiling the shell. AITune later compiles
only its compute child. This retains the pre-AITune measured strategy, not compiled timings. It does not establish
compatibility with arbitrary `compile_fn` transformations. Validate outputs, actual residency and serving performance;
retention and completed replanning are distinct finalization modes.

### Loading a saved profile

Pass `load_profile_from` explicitly to use public `ft.offload_from_profile` after shell insertion:

```python
prepared = prepare_flextensor_units(
    model,
    ft_config=cfg,
    name="aitune-example",
    load_profile_from="profiles/my-model",
)
model, manager = prepared.model, prepared.manager
```

This is an alternative setup in a fresh worker, not a second preparation of the model above. Merely setting
`profile_storage_dir` does not request profile loading. Profiles must match the wrapped model layout and workload;
profiles from unwrapped models are not automatically remapped. Compatible manual-shell profiles can be reused.
FT still controls any remaining startup; wait for inference before registration rather than assuming setup finished it.

## Registration and lifetime

`register_flextensor_units(managers, *, modules=None)` resolves each manager's final live public `offload_units()` after
it reaches inference. Omit `modules` for all shells or pass a subset of those public shells. The returned
`JITRegistration` view exposes the selected original **compute children**, reports and state counts. Register the complete
desired batch across managers once in a fresh worker.

The preparation helper requires an unprepared model and unused named manager. Prepare multiple models under distinct
names, finish startup for each, then register the complete desired batch once. Public state can reject a visibly active
manager but cannot prove that it was never initialized; do not pre-initialize or reuse a manager for another model.
Detected invalid policies, existing shells, non-executable container boundaries and cross-boundary aliases fail before
wrapping. If exact selector translation cannot preserve the policy, the helper raises rather than silently narrowing it.
Literal wildcard/normalization-sensitive names may require manual setup. Executable container subclasses are not
rejected merely because they are containers.

Provider setup and owner-setter failures can leave prepared state. They propagate without rollback; use a fresh worker.
Wrapping adds no weight copy; provider-owned placement and snapshots retain their normal costs.

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
for setup in manual helper; do
    python tests/functional/pytorch/jit/021_jit_flextensor.py --setup="$setup" --mode=eager
    python tests/functional/pytorch/jit/021_jit_flextensor.py --setup="$setup" --mode=deferred
    python tests/functional/pytorch/jit/021_jit_flextensor.py --setup="$setup" --mode=deferred --performance=enabled
    python tests/functional/pytorch/jit/021_jit_flextensor.py --setup="$setup" --mode=deferred --finalization=profile
    python tests/functional/pytorch/jit/022_jit_flextensor_fallback.py --setup="$setup"
done
python tests/functional/pytorch/jit/023_jit_flextensor_preparation.py

for placement in aitune external; do
    python tests/functional/pytorch/jit/020_jit_existing_prepared_units.py --mode=eager --device-management="$placement"
    python tests/functional/pytorch/jit/020_jit_existing_prepared_units.py --mode=deferred --device-management="$placement"
done
python -m pytest tests/unit/torch/jit/test_flextensor_preparation.py tests/unit/torch/jit/test_flextensor.py
```

Positive tests require all ten Inductor targets in diagnostic mode, normal correctness, actual per-graph profiling,
known/held-out numerical parity, real rolling storage under a strict reported weight-residency cap, one outer boundary
per traversal, and preserved live dispatch. The negative test requires a genuine compiler failure to preserve ordinary
eager fallback and provider residency without requesting a replan.

JIT023 compares public FT behavior with helper setup across name, class, parameter and exclusion policies, including
root/outside-unit parameters and fresh-process profile reloads. It compares storage-sharing partitions, not cross-process
addresses, and reported residency snapshots, not peak process memory. Dedicated helper unit tests also run without FT or
a GPU; they cover config/tensor identity, owner assignment, failure preflight and no hidden execution.

For the real pretrained all-48-block workload and manual video review, see the
[LTX 2.3 mini example](../../examples/LTX23OutpaintAITuneFlexTensor/README.md).
