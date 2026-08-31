---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Multi-GPU Integration
---

# Multi-GPU integration

AITune can tune inside an application-managed multi-process environment. It detects launcher and process-group state,
but it does not initialize, replace, or destroy `torch.distributed`. The application remains responsible for process
launch, process-group lifecycle, model sharding, collectives, and request coordination.

## Requirements

- Initialize the default PyTorch process group before starting AITune tuning.
- Bind each process to its local CUDA device before tuning.
- Invoke AOT tuning, or deferred JIT tuning and its next forward, on every rank.
- Use TorchInductor or Torch-TensorRT backend flavors for distributed modules. Unsupported backends are skipped
  cleanly by tune strategies.

AITune recognizes `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` from `torchrun`, and the corresponding
`OMPI_COMM_WORLD_*` variables from Open MPI. Launcher variables provide process identity and cache isolation, but live
tuning synchronization still requires an application-initialized PyTorch process group.

## Keep distributed logs readable

By default, `torchrun` sends stdout and stderr from every worker to the same console. Keep complete per-rank logs while
showing only local rank 0 interactively by configuring logging at the launcher:

```bash
torchrun \
  --log-dir logs \
  --tee 3 \
  --local-ranks-filter 0 \
  ...
```

`--tee 3` retains both streams from every worker under `--log-dir`, and `--local-ranks-filter 0` limits console output
without removing those per-rank files. On a multi-node run, local rank 0 from each node remains visible. This approach
does not require the application to suppress logging and preserves diagnostics from nonzero ranks. See the
[PyTorch `torchrun` logging documentation](https://docs.pytorch.org/docs/stable/elastic/run.html#logging) for redirect,
tee, filtering, and line-prefix options.

## Per-module execution modes

AITune classifies each tuning candidate independently; it does not classify every module as multi-GPU merely because
the application has multiple ranks. A rank-local module without distributed state uses the ordinary single-GPU
backend path, while a candidate containing DTensor parameters or buffers, or a `torch.distributed` module in its
subtree, requires a backend that supports multi-GPU execution. This allows ordinary and tensor- or context-parallel
parts to coexist in one model.

Some context-parallel implementations issue collectives from plain Python code without exposing distributed module
state. AITune detects native Diffusers context-parallel configuration automatically and selects its multi-GPU
execution path.

## AOT tuning

Call the ordinary AOT API from every rank after the application has initialized distributed execution and placed or
sharded the model. Each rank records and compiles its local shard. A backend is accepted only when it builds and passes
correctness checks on every rank.

Profiling strategies make one consistent choice across the job. They compare worst-rank results: minimum throughput and
maximum latency. Rank-local cache directories prevent concurrent artifact writes from colliding.

## JIT tuning

Multi-process JIT supports both eager and deferred modes when every rank executes the same modules in the same order
and reaches tuning readiness on the same forward. This is required because compilation uses the application's existing
default process group for synchronization. Divergent control flow or different sample readiness across ranks can
deadlock regardless of the selected mode.

Eager mode tunes automatically when the configured sample threshold is reached and is suitable for symmetric model
execution. Deferred mode lets the application choose the synchronized transition from recording to tuning, which is
useful for pipelines with repeated or variable module calls.

```python
import aitune.torch.jit.enable  # import before constructing the model
from aitune.torch import jit_config
from aitune.torch.jit import tune
from aitune.torch.jit.config import JITMode

jit_config.mode = JITMode.TUNE_DEFERRED

# Construct the model and let the application place or shard it.
# Run at least one representative forward on every rank to record samples.
model(*inputs)

tune.deferred()  # call on every rank; returns after every rank is armed
model(*inputs)   # the second forward tunes local shards on every rank
```

`tune.deferred()` uses the existing default process group as a rendezvous and synchronizes all ranks after arming
deferred tuning, immediately before the application continues to the second forward. AITune never creates a process
group.

## Placement and outputs

AITune automatically preserves the placement of modules detected as distributed, including modules containing DTensor
parameters and context-parallel modules recognized by enabled integrations.
Distributed tensor inputs also retain their existing placement. Ordinary inputs may still be moved to the resolved
rank-local device.

In multi-process runs, caches use rank-specific subdirectories and tuning-data and hardware-metrics filenames include
the rank. Logs include global and local rank. After tuning completes, AITune adds no synchronization to inference
forwards; the application and model continue to own inference collectives.

Distributed checkpoint packaging is not yet part of this integration. Live-tuned artifacts remain rank-local.

For complete applications, see the [LLM example](../../examples/LLM/README.md) for Transformers native tensor
parallelism and the [Flux example](../../examples/FLUX/README.md) for Diffusers context parallelism.
