---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: FlexTensor Integration
---

FlexTensor owns weight placement while AITune tunes the original offload-unit modules. Configure and prepare the model
through public FlexTensor APIs, finish provider startup, then register its public units for AITune JIT tuning.
Ordinary AITune correctness checks, performance profiling and backend selection remain enabled.

FlexTensor is optional for other AITune paths. Native tuning requires a source/development build containing the per-unit
compilation and replay fixes. The stock PyPI 0.5.0 release predates those fixes; a dependency on `flextensor>=0.5.0` alone
does not establish support. The maintained functional workflows and LTX example check for the native unit-index helper
and `flextensor.pre_compute.unit` operator at startup. Install the verified development wheel in the test environment
and preserve it when running workflows, for example with `uv run --no-sync`.

## Native offload units

Select the original model's executable blocks using an explicit `flextensor.OffloadConfig`. FlexTensor's
`offload_units()` identifies the original public units after discovery and profiling. The adapter registers those same
instances with `register_for_jit_tuning(..., device_management="external")`; it adds no module layer or parameter-path
translation. Keep original model selectors, attributes, hooks and state-dict names.

FlexTensor's native per-unit compilation and replay support manages transfers when AITune repeats a unit during warmup,
correctness checking and profiling. A repeated unit call is legitimate in this path; one unit invocation per whole-model
traversal is not a valid invariant. Application inference still runs through the exact root proxy returned by FlexTensor.
Do not call an isolated offloaded unit as a substitute for a complete model traversal.

The former `FlexTensorUnit`, `PreparedFlexTensor` and `prepare_flextensor_units` APIs have been removed. Prepare the original
model directly through public FT APIs. Profiles and checkpoints from the old wrapper layout are not automatically remapped.

## External compilation and explicit replan

This example assumes an uncompiled CPU `model` with executable `layers` and a representative CUDA input `x`.
Adapt the exact unit names, input shapes and memory policy to the workload. Leave room for inputs, activations and
compilation in addition to the provider's managed weight budget.

```python
import torch
import flextensor as ft

from aitune.torch import jit_config
from aitune.torch.backend import TorchInductorJitBackend, TorchInductorJitBackendConfig
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.flextensor import register_flextensor_units, replan_flextensor
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
    transfer_mode="allocation_block_transfer",
    profile_mode="view",
    profiling_iters=2,
    external_compile=True,
)
model = ft.offload(model.eval(), config=cfg, name="aitune-example")
manager = ft.get_offload_manager("aitune-example")

def run_forward():
    return model(x)  # Exactly one complete model traversal.

with torch.no_grad():
    for _ in range(manager.iters_before_inference):
        run_forward()
    if manager.phase.value != "inference":
        raise RuntimeError("FlexTensor startup did not complete")
    registration = register_flextensor_units([manager])
    run_forward()  # Record every representative shape/stage before deferred tuning.
    deferred()
    for _ in range(8):
        run_forward()
        if registration.all_tuned:
            break
    else:
        raise RuntimeError(f"Tuning did not complete: {registration.state_counts}")

    run_forward()  # Warm every selected serving route before measuring its plan.
    replan_flextensor(manager, registration, run_forward)
    result = run_forward()
```

A video pipeline can traverse its transformer many times per generation. Finish FT discovery and profiling before
registration, then record both stages before enabling deferred tuning. Warm all selected serving routes, including any
lazy compilation, before requesting a replan. The callback must execute **one** complete manager-model forward, not a
whole generation or denoising loop. The helper executes exactly the number of forwards requested by the public provider
API and requires a completed, applied replan with the same live unit identities and installed dispatch.

For a pipeline attribute, explicitly install the exact FT-returned proxy:

```python
pipe.transformer = ft.offload(pipe.transformer, config=cfg, name="ltx")
manager = ft.get_offload_manager("ltx")
```

Run startup and inference through that proxy or the normal pipeline, not an old transformer reference or a snapshot of
`manager.model`. Application-specific placement of parameters outside selected boundaries remains the caller's
responsibility; see the LTX example. The adapter does not change compiler limits. Observe effective process-wide values
when comparing runs because the provider or application may configure them.

## Retained view-profile strategy

The separate retained-profile workflow sets `profile_mode="view"`, `external_compile=False` and passes an identity
callback to activate FT's native compiled-unit path during preparation:

```python
cfg = ft.OffloadConfig(
    include_patterns=[f"layers.{index}" for index in range(len(model.layers))],
    profile_mode="view",
    external_compile=False,
)
model = ft.offload(model, config=cfg, name="aitune-example", compile_fn=lambda module: module)
manager = ft.get_offload_manager("aitune-example")
```

Finish startup, register the original units, record, tune and warm normally. Do not call `replan_flextensor` in this mode;
the helper rejects that request. The strategy retains its pre-AITune identity-compile profile rather than compiled
AITune timings. Retention and completed replanning are distinct finalization experiments. Arbitrary `compile_fn`
transformations are not qualified by this workflow.

### Loading a saved profile

Use public `ft.offload_from_profile` instead of `ft.offload` in a fresh worker:

```python
model = ft.offload_from_profile(
    model,
    "profiles/my-model",
    config=cfg,
    name="aitune-example",
    compile_fn=lambda module: module,  # Retained-profile mode only.
)
manager = ft.get_offload_manager("aitune-example")
```

For the external-compilation/replan mode, use its matching configuration and omit `compile_fn`. Merely setting
`profile_storage_dir` does not request profile loading. Profiles must match the native model layout, workload and
provider policy. Wait for the manager to reach inference before registration, including after profile loading.

## Registration and lifetime

`register_flextensor_units(managers, *, modules=None)` resolves each manager's public `offload_units()` in inference
phase. Omit `modules` to select all those units, or pass a subset of the exact public module objects. The returned
`JITRegistration.modules` contains those same original objects and exposes reports and state counts. Prepare multiple
models under distinct manager names, finish startup for each, then register the complete desired batch once per worker.

Keep manager/model identities, units, ownership and prepared forwarding fixed afterward. AITune installs its normal
router or tuned dispatcher on each original unit; replanning checks that dispatch and the underlying prepared forward
remain installed. Do not add constructor interception to an instance-registration worker. There is no reset, rollback,
re-registration or placement recovery API. Provider or callback failures propagate; stop that worker.

Ordinary AITune synchronous warmups, graph-break detection, autocast configuration, correctness checking, profiling,
dry-run, skip and eager fallback retain their existing behavior. An enabled performance gate may select its measured
eager baseline; that selection is different from a compilation failure. Eager fallback, skips and dry-run simulations
do not themselves request a replan.

`all_tuned` does not prove acceleration. Inspect selected backends, per-graph telemetry, outputs and measured performance.
`PerformanceValidationMode.DIAGNOSTIC` performs real eager/candidate profiling without rejecting a slower candidate;
it is useful for coverage tests, but does not establish a speedup.

## Compatibility and memory

The qualified backends and strategies are the existing
[external-placement candidates](jit_tuning.md#externally-owned-placement): exact eager/Inductor JIT classes and supported
Inductor modes/options. Other backends and CUDA-graph serving are not qualified by this integration. Native FT transfer
boundary operators are part of its compilation contract; there is no AITune-owned outer residency wrapper.

Separate the configured strategy fraction, reported managed weight residency and whole-process GPU memory. Reserve
known existing residents when choosing the strategy budget. `get_gpu_memory_usage()` counts transfer blocks and unmapped
tensors, not just `resident_units()`. Inputs, activations, workspace, allocator reserves and transient peaks require
separate measurement. A smaller fraction alone does not establish a strict process-wide limit. Cross-tree
parameter/storage aliases remain outside the placement contract.

## Verification

Use the development FT wheel described above and a fresh interpreter for each variant:

```bash
python tests/functional/pytorch/jit/021_jit_flextensor.py --mode=eager
python tests/functional/pytorch/jit/021_jit_flextensor.py --mode=deferred
python tests/functional/pytorch/jit/021_jit_flextensor.py --mode=deferred --performance=enabled
python tests/functional/pytorch/jit/021_jit_flextensor.py --mode=deferred --finalization=profile
python tests/functional/pytorch/jit/022_jit_flextensor_fallback.py

for placement in aitune external; do
    python tests/functional/pytorch/jit/020_jit_existing_prepared_units.py --mode=eager --device-management="$placement"
    python tests/functional/pytorch/jit/020_jit_existing_prepared_units.py --mode=deferred --device-management="$placement"
done
python -m pytest tests/unit/torch/jit/test_flextensor.py
```

These FT-specific scripts are temporarily excluded from automatic CI selection (`skip = true`) until a public
FT release includes the required fixes. Run them directly in an environment with the verified development wheel.

The positive native tests require all ten Inductor targets in diagnostic mode, normal correctness checks, actual
per-graph profiling, known/held-out numerical parity, rolling storage under a strict reported weight-residency cap,
unchanged original unit identities and parameter paths, and preserved dispatch. The negative test requires a genuine
compiler failure to preserve ordinary eager fallback, original hooks and provider residency without requesting a replan.
The adapter's unit tests run without FT or a GPU. The obsolete wrapper-selector preparation test has been removed.

For the real pretrained all-48-block workload, profile reload and manual video review, see the
[LTX 2.3 mini example](../../examples/LTX23OutpaintAITuneFlexTensor/README.md).
