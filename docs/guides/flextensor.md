---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: FlexTensor Integration
---

FlexTensor can retain ownership of weight offloading while AITune compiles its prepared offload units. The optional
`aitune.torch.jit.flextensor` adapter uses the public APIs in [FlexTensor 0.5.0](https://pypi.org/project/flextensor/0.5.0/).
Install that release separately; AITune does not require FlexTensor for other tuning paths:

```bash
pip install flextensor==0.5.0
```

## Startup and serving

Offload first, finish FlexTensor discovery and profiling, then register its final units once. Configure
`external_compile=True`, a block transfer mode (`"allocation_block_transfer"` or `"raw_block_transfer"`), and positive
`profiling_iters`. Do not pass `compile_fn`: AITune owns the per-unit compilation step in this flow.
Choose `max_gpu_mem_fraction` for the workload, leaving room for rolling transfer blocks and resident units. To
exercise real offloading, the complete weights must not fit in that budget.

The example assumes the caller provides an uncompiled CPU `model` with offloadable `layers.*` and a CUDA input `x`.
Adapt the selection pattern and representative inputs to the model. Each `run_forward()` invocation must execute
exactly one complete forward of this manager's model, including its normal provider hooks.

```python
import torch
import torch._dynamo.config as dynamo_config
from flextensor.config import OffloadConfig
from flextensor.offload_manager import OffloadManager

from aitune.torch import jit_config
from aitune.torch.backend import TorchInductorJitBackend, TorchInductorJitBackendConfig
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.flextensor import register_flextensor_units, replan_flextensor
from aitune.torch.jit.tune import deferred
from aitune.torch.tune_strategy import OneBackendStrategy

# Choose a finite budget for this workload; retain any larger caller setting.
for name in ("recompile_limit", "accumulated_recompile_limit", "cache_size_limit", "accumulated_cache_size_limit"):
    if hasattr(dynamo_config, name):
        setattr(dynamo_config, name, max(getattr(dynamo_config, name), 256))

jit_config.mode = JITMode.TUNE_DEFERRED
jit_config.device = "cuda"
jit_config.min_samples = 1
jit_config.batch_axis_required = False
jit_config.detect_graph_breaks = False
jit_config.strategy = OneBackendStrategy(
    TorchInductorJitBackend(TorchInductorJitBackendConfig(fullgraph=True, mode="default"))
)
manager = OffloadManager("aitune-example")
model = manager.offload(
    model.eval(),
    OffloadConfig(
        include_patterns=["layers.*"],
        transfer_mode="allocation_block_transfer",
        profiling_iters=2,
        external_compile=True,
    ),
)

def run_forward():
    return model(x)

with torch.no_grad():
    for _ in range(manager.iters_before_inference):
        run_forward()
    assert manager.phase.value == "inference"

    registration = register_flextensor_units([manager])
    for _ in range(jit_config.min_samples):
        run_forward()  # Record complete model calls before arming deferred tuning.
    deferred()
    for _ in range(8):  # Bound startup; do not wait indefinitely for missing routes.
        run_forward()
        if registration.all_tuned:
            break
    else:
        raise RuntimeError(f"Tuning did not complete: {registration.state_counts}")

    run_forward()  # Warm the selected serving route before timing its offload plan.
    replan_flextensor(manager, registration, run_forward)
    result = run_forward()  # Serve using the same model and provider-managed residency.
```

Warm every representative shape and route before replanning. The example's `all_tuned` requirement is a startup
policy, not proof of acceleration: verify selected backends, numerical correctness, and performance before deployment.
An actual selected `TorchEagerBackend` is a tuned result and can require replanning because AITune routing affects
timing. Skipped targets, native eager fallback, and dry-run simulations do not by themselves request a replan. Use
`registration.state_counts` and `registration.reports` to distinguish these outcomes rather than waiting for
`all_tuned` in an application that permits fallback.

## Adapter contract

`register_flextensor_units(managers, *, modules=None)` discovers the managers' public `offload_units()` and returns the
ordinary `JITRegistration` from one external-placement registration. Omit `modules` to select all discovered units, or
pass a subset consisting only of current public offload units. Supply the complete desired set across all managers
together, after every manager reaches inference and before any AITune registration in that worker. Keep the model
trees and provider ownership fixed afterward; selected roots must have disjoint ownership trees.

`replan_flextensor(manager, registration, run_forward)` is an explicit, synchronous startup step after that manager's
selected targets have settled and their real tuned routes have been warmed. It checks the public replan availability,
requests a replan when a real tuning result requires one, and invokes `run_forward` exactly the positive number of
times returned by the provider. Completion requires public status `state == "done"`, `applied is True`, and preserved
AITune forward routing. Call it separately for each participating manager with that manager's whole-model callback.

The callback must not be a unit-only call, a multi-forward pipeline or denoising loop, or a CUDA-graph replay. The
provider counts whole-model forwards to advance its profiling lifecycle. The adapter installs no root coordinator
hooks and does not advance replanning in background serving calls. Treat registration, tuning-finalization, callback,
or replan errors as terminal startup failures and start a fresh worker; there is no reset, rollback or re-registration
flow.

## Placement and compiler scope

FlexTensor continues controlling source residency during recording, compilation, warmup, replanning and serving.
AITune protects the existing source tree from its own movement paths; it can still move recorded inputs and owned
artifacts. The caller provides memory headroom, a fixed model tree, and a finite process-wide
[TorchDynamo budget](jit_tuning.md#compiler-budget). AITune does not reserve or restore compiler capacity.

Use the exact `TorchEagerBackend` or `TorchInductorJitBackend` classes. Inductor mode must be `None`, `"default"` or
`"max-autotune-no-cudagraphs"`, without custom options. Configure explicit candidates with the exact `OneBackendStrategy`,
`FirstWinsStrategy`, `MinLatencyStrategy`, `MaxThroughputStrategy` or `LatencyBudgetStrategy` class; the default strategy
includes unsupported external candidates. See [Externally owned placement](jit_tuning.md#externally-owned-placement)
for source-identity protection and aliasing constraints.

The [FlexTensor functional test](https://github.com/ai-dynamo/aitune/blob/main/tests/functional/pytorch/jit/021_jit_flextensor.py) pins the same provider release
and covers this startup sequence with real offloading and compilation. Native GPU execution is pending CI validation;
the added adapter and test do not establish a validated provider/GPU combination by source inspection alone.

From an AITune checkout, run each GPU variant in its own fresh interpreter:

```bash
python tests/functional/pytorch/jit/021_jit_flextensor.py --mode=eager
python tests/functional/pytorch/jit/021_jit_flextensor.py --mode=deferred
```

The script uses roughly 704 MiB of model weights with a smaller FlexTensor placement budget, plus compiler workspace.
It checks numerical results, real Inductor selection, applied replanning and compiled dispatch after the rebuild.
