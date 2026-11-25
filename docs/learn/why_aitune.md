---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Why AITune"
---

Accelerating a PyTorch application on NVIDIA GPUs often means answering a series of questions. Where does inference
spend time? Which parts can run independently? Which compiler, runtime, or optimization works for each part? Does the
result meet the application's performance and quality requirements?

AITune makes this work repeatable. The same pipeline can contain modules with different input shapes,
execution patterns, and GPU placement. AITune observes the running application, isolates selected module calls, and
gives tuning strategies a common way to evaluate different implementations. The application can then run with the
selected implementations in place.

## Start with how the application runs

An image generation pipeline might encode a prompt once, call a denoiser repeatedly, then decode the result. AITune
inspection shows which modules ran, how often they ran, and where time went. Its optional call tree adds calling
relationships, tensor flow, and observed uses of weights outside a module. That context helps engineers decide where
to tune and which dependencies to preserve.

If the denoiser takes 80 ms of a 100 ms sequential request, halving its time brings the request to 60 ms when the rest
of the request takes the same time. That is a 2× speedup for the denoiser and about 1.67× for the request. Measure the
whole pipeline to see the actual benefit, including repeated calls, overlapping work, and any changes in overhead.

## Extend the system in your own code

A **backend** owns how an implementation is built and run. A **strategy** owns how candidates are evaluated and
selected. Both are plugin interfaces. A team can implement either one in its own package, keep the full logic there,
and pass it to the existing tuning workflow. The extension only needs to satisfy the applicable interface, so teams
can try ideas and integrate private runtimes without waiting for an upstream change. The same boundaries give
contributors a focused place to implement and review features that belong in AITune.

The application supplies representative input through its dataloader or inference calls, defines the performance goal,
and validates the tuned pipeline on its target workload. AITune handles the shared recording, tuning, and execution
workflow. See [Core Concepts](concepts.md) and the [Architecture](../guides/advanced/architecture.md).
