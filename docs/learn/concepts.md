---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Core Concepts"
---

## Scenario

An image generation application encodes a prompt, calls a denoiser repeatedly, and decodes the result into an image.
AITune can tune the denoiser and decoder independently while the Python pipeline still controls when each one runs.
The example below follows a request through that process.

## Inspection and tuning boundaries

**Inspection** runs the application with example requests to see which PyTorch modules execute, how often they run,
and how much time they take. With call tree recording enabled, you can also see which calls lead to each module,
where tensors come from, and which weights code outside a module uses. A denoiser that runs repeatedly may be a
good place to start because even a small improvement per call can add up across the request.

A **tuning boundary** is the module you choose to wrap and the work inside its forward call. Wrapping the denoiser
lets AITune record its calls in the running pipeline and replay them separately to evaluate candidate implementations.
The scheduler still decides when to call it, and the decoder still turns the denoised result into an image. The
decoder can have its own tuning boundary and use a different backend. When choosing a boundary, use the observed
module relationships to account for dependencies such as weights shared with code outside the module.

## Representative inputs and graph specifications

**Representative inputs** are requests that exercise the conditions you intend to serve. In the explicit workflow,
you supply them through a dataset or dataloader and run your usual inference callable. For image generation, those
requests might include prompts, image sizes, and generation options. AITune records the arguments that actually
reach each wrapped module. The denoiser receives intermediate tensors and conditioning prepared by the pipeline.

A **graph specification** describes one observed form of a module call. It records the input and output structure,
tensor metadata such as shapes and dtypes, and observable behavior such as supported input mutations. For example,
if the pipeline calls the same denoiser with and without an optional conditioning tensor, those input structures
can produce two graph specifications. AITune evaluates candidates separately for each specification. Variations in
a dynamic dimension, such as batch size, can be represented as a range within a single specification.

### Check recorded samples and input shape limits

Use your dataloader to cover the call patterns and shapes that matter to your application. With correctness checks
enabled, AITune runs the candidate on recorded samples and checks that outputs have the expected shapes and that
output values are finite. It also checks the state of inputs after the call, including tensor shapes and non-tensor
values that the original module changed.

For graphs with detected dynamic axes, AITune also runs the candidate at the recorded minimum and maximum input
shapes and checks the output shapes. If the denoiser's recorded batch range is 1 to 4, this exercises batches of
1 and 4. Compare the tuned and original pipelines on representative requests to assess numerical agreement and
image quality across the workload you plan to serve.

## Backends and strategies

A **backend** owns how a module implementation is built and run. For the denoiser, a TensorRT backend might build
and execute an engine, while a Torch Inductor backend might compile the module through PyTorch. Each backend
handles its own build and runtime requirements. A candidate is a backend with a particular configuration, so a
strategy can evaluate different configurations of the same backend if configured.

A **strategy** owns how candidates are evaluated and which one is selected. An interactive image generator might
prioritize latency, while a service processing a queue of requests might prioritize images per second. The same
candidates can lead to different selections under those objectives. Other strategies can try candidates in order
or use a specified backend. Once selected, the backend serves subsequent calls that match the tuned graph.

Both are plugin interfaces. You can implement the full backend or strategy in your application package and pass
it to the normal tuning workflow. A custom backend can run under a built-in strategy, and a custom strategy can
evaluate built-in backends. See the [Backend Catalog](../guides/backends/overview.md) for built-in options and
[Extending AITune](../guides/advanced/extending_aitune.md) for the plugin interfaces.

## When to use AOT or JIT

With **ahead-of-time (AOT) tuning**, you explicitly select and wrap modules and start tuning with representative
inputs. This gives you control over the tuning run and lets you prepare checkpoints or supported deployment
artifacts before serving requests.

With **just-in-time (JIT) tuning**, AITune intercepts eligible modules and collects samples as the existing
application runs. In automatic mode, tuning starts when the configured readiness conditions are met. These can include a minimum
number of calls and detection of a dynamic axis. In deferred mode, the application signals when tuning may start.
See [JIT tuning modes](../guides/jit_tuning.md#tuning-modes).

| Feature | Ahead-of-time (AOT) | Just-in-time (JIT) |
| --- | --- | --- |
| How tuning starts | Call `ait.tune()` explicitly. | Enable JIT and let tuning start during inference, or signal when to start in deferred mode. |
| Module selection | You choose which modules to wrap and tune. | AITune discovers eligible modules during inference. |
| Representative inputs | Supply a dataset or dataloader and configure batch sizes. | AITune records calls made by the application. |
| Dynamic shapes | Record representative shapes or provide explicit bounds where supported. | Record multiple shapes before tuning to detect dynamic axes. |
| Candidate profiling | Profiling strategies measure candidates. Supported strategies can search for the maximum batch size. | Profiling strategies measure candidates using recorded calls. Maximum batch size discovery is disabled. |
| Strategy configuration | Assign a strategy to individual modules when needed. | Configure a shared strategy. Default backend candidates are resolved for each module. |
| Reuse in another process | Save an AITune checkpoint and load it into a compatible model or pipeline. | Each new process tunes again. The cache stores build artifacts and logs. |

The workflow determines when the application starts tuning. The backend's build mode determines how the
implementation is prepared. An explicit AOT run can use a backend that compiles just in time.

## Save a checkpoint or export a deployment artifact

An **AITune checkpoint** saves the tuning state so you can load it into a compatible Python model or pipeline.
For example, a new inference process can construct the same image generation pipeline and load the tuned denoiser
and decoder. A **deployment artifact** is a backend-specific result, such as a TensorRT engine, that a supported
serving runtime can load directly. Choose based on whether you need to run the Python pipeline or serve a compiled
component. See [Deployment Overview](../guides/deployment/deployment.md).

Continue with the [Quick Start](quick_start.md) to tune a pipeline or the
[Architecture](../guides/advanced/architecture.md) to see how the components fit together.
