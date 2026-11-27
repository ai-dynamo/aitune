---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: FlexTensor Integration
---

FlexTensor owns weight placement while AITune tunes its original offload-unit modules. Prepare the model with public
FlexTensor APIs, finish discovery and profiling, then register those units for JIT tuning. The adapter adds no module
layer or parameter-path translation.

Install `flextensor>=0.5.0` separately; FlexTensor remains optional for other AITune workflows.

## Retain the initial offload profile

Start with this workflow: **prepare → finish FT startup → register → record → tune → warm → serve**. It retains FT's
initial weight-transfer strategy while AITune selects a backend for each recorded graph. It does not measure a new
strategy from compiled timings.

The example assumes an uncompiled CPU `model` whose `layers` are complete executable blocks, and a representative
CUDA input `x`. Adapt the unit names, inputs and memory fraction to your model. Place any parameters outside the selected
units as required by the application, and reserve memory for inputs, activations and compilation.

```python
import torch
import flextensor as ft

from aitune.torch import jit_config
from aitune.torch.backend import TorchInductorJitBackend, TorchInductorJitBackendConfig
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.flextensor import register_flextensor_units
from aitune.torch.jit.tune import deferred
from aitune.torch.tune_strategy import OneBackendStrategy

jit_config.mode = JITMode.TUNE_DEFERRED
jit_config.device = "cuda"
jit_config.min_samples = 1
jit_config.batch_axis_required = False
jit_config.strategy = OneBackendStrategy(
    TorchInductorJitBackend(TorchInductorJitBackendConfig(fullgraph=True, mode="default"))
)

cfg = ft.OffloadConfig(
    include_patterns=[f"layers.{index}" for index in range(len(model.layers))],
    max_gpu_mem_fraction=0.5,  # Choose a weight-planning budget for this workload.
    transfer_mode="allocation_block_transfer",
    profile_mode="view",
    profiling_iters=2,
    external_compile=False,
)
model = ft.offload(model.eval(), config=cfg, name="aitune-example", compile_fn=lambda module: module)
manager = ft.get_offload_manager("aitune-example")
representative_inputs = [x]  # Include each serving shape or stage before tuning.


def run_forward(value=x):
    return model(value)  # Always use the exact root proxy returned by FT.


with torch.no_grad():
    for _ in range(manager.iters_before_inference):
        run_forward()
    if manager.phase.value != "inference":
        raise RuntimeError("FlexTensor startup did not complete")

    registration = register_flextensor_units([manager])
    for value in representative_inputs:
        run_forward(value)
    deferred()
    for _ in range(8):
        for value in representative_inputs:
            run_forward(value)
        if registration.all_tuned:
            break
    else:
        raise RuntimeError(f"Tuning did not complete: {registration.state_counts}")

    for value in representative_inputs:
        run_forward(value)  # Warm each route, including lazy compilation.
    result = run_forward()
```

The identity `compile_fn` activates FT's native compiled-unit path without transforming the modules. Native per-unit
replay manages transfers during AITune's repeated warmup, correctness and profiling calls. Application inference still
uses the complete managed model; calling an isolated offloaded unit is not a substitute for a full traversal.
Do not call `replan_flextensor` in retained-profile mode. Arbitrary `compile_fn` transformations are not qualified.

For a pipeline attribute, assign the exact returned proxy to its owner, for example
`pipe.transformer = ft.offload(pipe.transformer, config=cfg, name="ltx", compile_fn=lambda module: module)`.
Run startup and inference through the updated pipeline, not an old transformer reference or a snapshot of `manager.model`.
A video generation may contain many transformer forwards: record and warm every serving stage before benchmarking.

### Load a saved profile

In a fresh worker, replace the `ft.offload` call above with:

```python
model = ft.offload_from_profile(
    model.eval(),
    "profiles/my-model",
    config=cfg,
    name="aitune-example",
    compile_fn=lambda module: module,
)
```

Continue the same manager lookup, startup check, registration and tuning flow. Merely setting `profile_storage_dir`
does not load a profile. Saved profiles must match the native model layout, workload and provider policy; changing a
budget while reusing a concrete profile does not create a newly planned experiment. Incompatible model/checkpoint
layouts are not automatically remapped.

## Advanced: explicitly replan after tuning

Choose this mode **before FT preparation in a fresh worker**. Relative to the flow above:

1. Set `external_compile=True` and omit `compile_fn` from `ft.offload` or `ft.offload_from_profile`.
2. Complete the same startup, registration, recording, tuning and route warmup.
3. Before serving, request a new strategy using a representative full-model forward:

```python
from aitune.torch.jit.flextensor import replan_flextensor

with torch.no_grad():
    replan_flextensor(manager, registration, run_forward)
```

Each callback invocation must execute exactly **one** complete manager-model forward, not a whole video generation or
denoising loop. The helper executes the positive count requested by FT and requires `state="done"`, `applied=True`,
unchanged ownership/prepared forwards and preserved AITune dispatch. Select representative inputs for the plan;
warming every route does not make a single replan input representative of all shapes or stages.

## Registration, safety and supported configurations

`register_flextensor_units(managers, *, modules=None)` resolves the final public `manager.offload_units()` in inference
phase and registers those same original objects with `device_management="external"`. Omit `modules` to select all units,
or pass a subset of those exact objects. Finish startup for every supplied manager, then register the complete desired
batch once per worker. The returned `JITRegistration` exposes the original modules, live reports and state counts.

Keep manager/model identities, units, ownership and prepared forwards fixed afterward. Do not combine instance
registration with automatic constructor interception. Setup, provider and callback failures propagate; use a fresh
worker. There is no reset, rollback, re-registration or placement recovery API. Cross-tree parameter/storage aliases
remain outside the placement contract.

The qualified backends and strategies are the existing
[external-placement candidates](jit_tuning.md#externally-owned-placement): exact eager/Inductor JIT classes and supported
Inductor modes/options. Other backends and CUDA-graph serving are not qualified. The caller configures finite compiler
limits and memory headroom; the adapter does not reserve or restore compiler capacity. Observe effective limits because
the provider or application may change them.

Ordinary correctness checks, graph-break detection, profiling, skip, dry-run and eager fallback retain their behavior.
An enabled performance gate may select its measured eager baseline. `all_tuned` alone proves neither compiled execution
nor acceleration; inspect actual backend selections and outputs. Diagnostic performance validation measures both
candidates without rejecting a slower one. Eager fallback, skips and dry-run simulations do not themselves request a replan.

Separate the planning fraction, reported managed weight residency and whole-process GPU memory. FT's
`get_gpu_memory_usage()` includes transfer blocks and unmapped tensors; activations, other components, compiler workspace,
allocator reserves and transient peaks need separate measurement. A configured fraction is not a strict process-wide cap.

## Validation and runnable checks

The adapter tests cover native module/parameter identity, prepared forwards, registration preflight and completed
replan/dispatch contracts. GPU functional tests additionally check actual backend profiling, known/held-out numerical
parity on their test models, reported offloading limits and genuine compiler-failure fallback. These checks do not
establish numerical equivalence for every pretrained workload: validation found compiled LTX held-out outputs that
failed strict `atol=rtol=1e-2` comparison with resident eager. Built-in AITune checks validate finiteness, shapes and input
metadata; they are not eager-output allclose checks. No general speedup or perceptual-quality acceptance is claimed.

From the repository root, run each functional command in a fresh interpreter:

```bash
python -m pytest tests/unit/torch/jit/test_flextensor.py
python tests/functional/pytorch/jit/021_jit_flextensor.py --mode=deferred --finalization=profile
python tests/functional/pytorch/jit/021_jit_flextensor.py --mode=eager
python tests/functional/pytorch/jit/021_jit_flextensor.py --mode=deferred
python tests/functional/pytorch/jit/021_jit_flextensor.py --mode=deferred --performance=enabled
python tests/functional/pytorch/jit/022_jit_flextensor_fallback.py

for placement in aitune external; do
    python tests/functional/pytorch/jit/020_jit_existing_prepared_units.py --mode=eager --device-management="$placement"
    python tests/functional/pytorch/jit/020_jit_existing_prepared_units.py --mode=deferred --device-management="$placement"
done
```

Unit tests need neither FT nor a GPU; the functional workflows need CUDA and `flextensor>=0.5.0` with its native
per-unit compilation/replay support. See the [LTX example](../../examples/LTX23OutpaintAITuneFlexTensor/README.md) for
pinned pretrained inputs, matched benchmark commands and held-out output inspection.
