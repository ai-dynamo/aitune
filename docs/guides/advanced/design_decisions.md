---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Design Decisions"
---

Start with [Why AITune](../../learn/why_aitune.md) for the motivation and
[Core Concepts](../../learn/concepts.md) for an introduction to tuning boundaries, backends, strategies, and the
AOT and JIT workflows. This guide explains the reasons behind those choices.

AITune's design makes it possible to optimize individual parts of an inference application and add custom backends
and tuning strategies in the user's environment. A backend defines how an implementation is built and run, such as
building and running a TensorRT engine. A tuning strategy defines how candidates are evaluated and selected.
For example, it can compare candidates and select the one with the lowest measured latency, or try them in order
and accept the first that passes the configured correctness and
performance checks. Teams can implement either interface in their own code and use it in the existing tuning workflow.

## Observe execution before choosing tuning boundaries

Runtime behavior exposes which modules actually run, their inputs, and relationships such as a caller
reading a module's weights. When wrapping a module selected from a call tree, AITune can keep weights on the device
if it observed code outside the module using them.

Inspection takes an additional application run, and its findings reflect the inputs supplied.

## Tune at module boundaries

AITune can isolate a module call and evaluate it with actual application inputs. Different modules can use
different backends while the surrounding application keeps its existing execution flow.

## Keep backends and tuning strategies separate

Backends build and run implementations. Tuning strategies evaluate candidates and select an implementation.
Teams can develop and combine both locally, which shortens the path from an idea to a measured result and keeps
upstream contributions focused.

See the [Backend Catalog](../backends/overview.md) for supported implementations and
[Tuning Strategies](../tune_strategies/tune_strategies.md) for evaluation and selection options.

## Use the application's inputs and measure on the target hardware

Inputs recorded from the application provide realistic module calls. Measurements on the target hardware
help strategies select candidates against the chosen performance objective.

More samples and candidates take longer to evaluate. The useful balance depends on the input coverage
and tuning time the application needs. Results can change with the inputs, configuration, or hardware.

The [Tuning Workflow](tuning_workflow.md) describes recording inputs, evaluating candidates, and activating backends.

## Support explicit tuning and tuning during inference

With [ahead-of-time (AOT) tuning](../aot_tuning.md), teams preparing artifacts can control the tuning run. With
[just-in-time (JIT) tuning](../jit_tuning.md), teams integrating into an existing process can capture inputs during
inference and decide when to tune. Both use the module, strategy, and backend contracts.

Explicit tuning requires a preparation step with representative inputs. Tuning during inference
collects inputs from normal calls, with recording and tuning work taking place in the running process.

## Provide defaults while keeping tuning configurable

AITune provides default strategies and backend candidates so users can start tuning without configuring every
detail. Users can then choose the desired outcome: minimize latency, maximize throughput, or maximize throughput
within a latency limit.

In the AOT workflow, users can pass a strategy to selected modules. In JIT, they configure the strategy through the
[JIT configuration](../jit_tuning.md#configuration). Both accept `resolve_strategy()`, which selects compatible
backend candidates according to the requested objective and compilation mode. For example, `compilation="aot"`
restricts candidates to ahead-of-time
backends. Users can also supply an explicit strategy with their own backend candidates and configurations.

The tuning workflow and backend compilation mode are separate choices: tuning during application execution can
still prepare an AOT backend. See [Tuning Strategies](../tune_strategies/tune_strategies.md) for configuration options.

The [AOT configuration guide](../aot_tuning.md#configuration-options) and
[JIT configuration guide](../jit_tuning.md#configuration) describe the remaining workflow settings.

## Application responsibilities

AITune assumes that one thread executes a given model's calls at a time. Recording, tuning, and runtime dispatch
rely on this assumption when managing model state. The application must serialize calls to the same model rather
than rely on AITune to add locks around concurrent calls.

The application owns the request and its performance and quality targets. It also owns distributed setup, model
sharding, and the inference collectives that the model requires. AITune coordinates tuning within that environment
and evaluates module candidates compatible with it. See [Multi-GPU Integration](../multi_gpu.md) and
[Deployment Overview](../deployment/deployment.md).
