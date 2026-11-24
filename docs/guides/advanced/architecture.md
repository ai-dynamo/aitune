---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Architecture"
---

AITune runs alongside your model in your Python environment. Your code calls the model or pipeline with
representative inputs. AITune observes those calls and uses them to tune selected modules. During inference,
your code calls the same model or pipeline and the wrapped modules run their selected backends.

The diagram shows the explicit AOT workflow. You can use built-in backends and strategies or provide your
own implementations of either interface.

<img src="../../assets/aitune_architecture.svg" alt="User code calls the model in the user's Python environment. AITune inspects execution, records calls to wrapped modules, and tunes each graph. User-provided backend and strategy implementations can take part in tuning. The same user code then runs inference with the selected backends." width="840" />

## From an application call to an isolated module

1. **Inspect the application.** AOT inspection runs user-provided inputs to find executed modules and measure their
   calls. The optional call tree captures caller relationships, tensor sources, and observed external accesses to
   weights. For example, if Python code outside the selected denoiser reads one of its weights, call tree information
   can help the wrapper preserve that weight during offload. JIT uses interception to discover eligible modules as
   the application runs.
2. **Record the boundary.** The wrapper runs the original forward and captures the samples and metadata needed to
   replay it independently. It groups observations by input specification and records input and output structures and
   the supported effects needed to reconstruct behavior at the boundary.
3. **Evaluate implementations.** A strategy receives the original module, graph specification, and sample store.
   It decides which candidates to build, which checks and measurements to apply, and which implementation to
   return. A backend receives the same module-level context and owns compiler or runtime calls and resource
   management. Your code can provide a backend, a strategy, or both.
4. **Run the selected backend.** The tuned wrapper uses incoming call metadata to dispatch to the selected backend
   for the matching graph. The surrounding pipeline still manages the inference request, including calls between
   modules. When AITune checks an incoming call against the tuned graph specifications, it raises an error if none match.

## Where to find the implementation

| Responsibility | Code |
| --- | --- |
| Explicit inspection, call tree, and wrapping | `aitune/torch/inspecting/` and `aitune/torch/call_tree/` |
| AOT tuning orchestration and JIT interception | `aitune/torch/tuning.py` and `aitune/torch/jit/` |
| Recording, graph specification, and runtime dispatch | `aitune/torch/module/recording_module.py`, `graph_spec.py`, and `tuned_module.py` |
| Strategy and backend interfaces | `aitune/torch/tune_strategy/tune_strategy.py` and `aitune/torch/backend/backend.py` |
| Validation, tuning measurements, and application profiling | `aitune/torch/task/` and `aitune/torch/performance/` |
| Persistence and serving integrations | `aitune/torch/checkpoint/`, `aitune/dynamo/`, and `aitune/triton/` |

For details on the call tree and weight relationships, see [AOT Inspection](../aot_inspect.md).
Tuning reports describe candidate evaluation and selection. Application profiling shows how each tuned or untuned
component contributes to request time. See [Observability](../../learn/observability.md),
[Extending AITune](extending_aitune.md), and [Design Decisions](design_decisions.md).
