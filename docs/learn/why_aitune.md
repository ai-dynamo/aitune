---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Why AITune"
---

Moving a trained model into production means meeting latency, throughput, GPU memory, and output quality requirements
on the hardware that will serve requests. New compilers, runtimes, and optimization techniques offer ways to improve
performance, but evaluating them has a cost. Each integration requires engineering work to prepare inputs, adapt
execution, validate results, and maintain compatibility as the application changes. Teams may keep an existing
implementation because the migration and ongoing maintenance effort outweigh the performance benefit they expect.

AITune reduces the cost of exploring and adopting these technologies. It provides a shared workflow for evaluating
supported backends with actual module inputs and running selected implementations through existing application code.
Teams can try another backend without building and maintaining a separate inference integration for each option.

This workflow addresses practical questions that arise when optimizing an application:

- Which modules account for enough inference time to make optimizing them worthwhile?
- Which backend and configuration work well for each module's computation, input shapes, and memory requirements?
- How can intermediate module inputs be collected for compilation, testing, and performance measurement?
- Will the optimized implementation support the application's input variation and preserve the required behavior?
- How can it run inside the existing model or pipeline without rewriting the inference flow for each backend?
- How can the team save the results and repeat the process when the model, inputs, or hardware change?

AITune observes execution, records inputs to selected PyTorch modules, evaluates configured backend candidates,
and runs the selected implementations through the application's existing module calls. This includes ahead-of-time
backends such as TensorRT.

## Start with how the application runs

An image generation pipeline typically contains a text encoder, a denoiser, and a decoder. The text encoder converts
the prompt into conditioning tensors. The denoiser progressively removes noise from a latent representation, and
the decoder converts the resulting latent representation into an image.

During a request, the pipeline encodes the prompt, then repeatedly calls the denoiser according to a scheduler that
controls the denoising steps. After the final step, it calls the decoder. These components have different input
shapes, memory requirements, and call frequencies, so they can benefit from different optimizations.

AITune inspection shows which modules ran, how often they ran, and how much time each component contributes to the
request. This helps identify where optimization will have the greatest effect. Its optional call tree adds calling
relationships, tensor flow, and observed uses of weights outside a module. That context helps engineers preserve
dependencies when choosing which modules to tune.

If the denoiser takes 80 ms of a 100 ms sequential request, halving its time brings the request to 60 ms when the rest
of the request takes the same time. That is a 2× speedup for the denoiser and about 1.67× for the request. Measure the
whole pipeline to see the actual benefit, including repeated calls, overlapping work, and any changes in overhead.

## Evaluate alternatives with actual module inputs

Modules often receive intermediate tensors and state prepared by the surrounding application. AITune records those
calls while the application runs, so teams can evaluate candidates without writing a separate input extraction
script for each module. Graph specifications track observed call structures, tensor metadata, and supported changes
to mutable inputs. Dynamic shape tracking captures observed ranges; supported explicit bounds can describe the
intended input range.

Tuning strategies use this recorded context to build and check candidates. Profiling strategies measure performance
on the target hardware and select according to their objective. Each module can use a different backend or
configuration, so the application can combine optimizations suited to its different components.

## Use optimized implementations in the existing application

An ahead-of-time engine normally requires code to build, load, and execute it. AITune backends handle these steps
behind the module interface. After tuning, calls to a wrapped module run the selected implementation while the
surrounding Python pipeline continues to control the request.

AITune checkpoints let teams restore tuning state into a compatible model or pipeline in another process. Supported
backends can also export deployment artifacts. When the model, representative inputs, or hardware change, teams
can reuse the tuning workflow and repeat evaluation rather than rebuild their integration code.

## How AITune compares with torch.compile

[`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) captures computation as the
application runs and compiles it through a selected backend. It can adapt to observed execution paths and input
changes through guarded recompilation and dynamic shapes. It supports multiple backends, and compiler modes can
autotune implementation choices within a backend. Like AITune, it lets optimized computation run through existing
Python calls.

With direct `torch.compile` use, the team chooses which functions or modules to compile and which backend and
configuration to use. Comparing alternatives, measuring their effect on application performance, and managing
integrations with other acceleration technologies require additional work.

AITune provides a shared workflow for that work. It records actual module inputs and uses tuning strategies to
evaluate configured backend candidates, check their behavior, and select implementations for individual modules.
Its Inductor JIT backend uses `torch.compile`, so teams can evaluate it alongside supported AOT backends and other
optimizations through the same workflow. Profiling strategies compare candidate performance; compiler autotuning
still happens within the backends that provide it.

AITune also coordinates building, saving, restoring, and running supported AOT implementations in the existing
application. PyTorch provides [AOT compilation facilities](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_aot_compile.html)
as well. AITune's benefit is the common evaluation and integration workflow across its supported backends, reducing
the code teams maintain when trying another technology. Performance gains depend on the configured candidates,
representative inputs, and target hardware.

## Extend the system in your own code

A team can add a backend for a private runtime or a strategy with its own evaluation and selection rules without
replacing the shared workflow. Both can live entirely in the team's package and use the same recorded module calls
as built-in implementations. This lets teams compare new approaches with existing backends without maintaining a
separate tuning script for each integration or waiting for an upstream change. The same interfaces give contributors
a focused place to implement and review features that belong in AITune.

## Validate the complete application

The application supplies representative inputs through its dataloader or inference calls and defines the performance
goal. AITune's built-in correctness checks help reject candidates with incompatible output shapes, non-finite
outputs, or unexpected changes to inputs. Teams still validate numerical agreement, output quality, and end-to-end
performance on the requests they intend to serve. They also own request scheduling, concurrency, distributed setup,
and production operations.

Continue with [Core Concepts](concepts.md) to learn the terms, interfaces, and workflows, then read
[Design Decisions](../guides/advanced/design_decisions.md) for the rationale and tradeoffs behind AITune's design.
The [Architecture](../guides/advanced/architecture.md) explains how the components fit together.
