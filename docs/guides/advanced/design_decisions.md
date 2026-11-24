---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Design Decisions"
---

AITune's design makes it possible to optimize one part of an inference application and to add new acceleration
mechanisms or selection policies inside the user's environment.

| Decision | Why we made it | Practical tradeoff |
| --- | --- | --- |
| **Observe execution before choosing tuning boundaries.** | Runtime behavior exposes which modules actually run, their inputs, and relationships such as a caller reading a module's weights. When wrapping a module selected from a call tree, AITune can keep weights on the device if it observed code outside the module using them. | Inspection adds work up front and describes the paths exercised by the supplied workload. |
| **Tune at module boundaries.** | AITune can isolate a module call and evaluate it with actual application inputs. A repeatedly called denoiser and a decoder can use different backends. | A larger boundary can expose more work to a compiler. A smaller one can isolate unsupported behavior, but may miss opportunities to optimize across the boundary. Dependencies such as shared weights still matter. |
| **Put mechanism and policy behind separate plugin interfaces.** | Backend code owns the complete acceleration mechanism. Strategy code owns the complete selection policy. Teams can develop and combine both locally, which shortens the path from an idea to a measured result and keeps upstream contributions focused. | Plugin authors own compatibility, lifecycle, validation policy, and persistence behavior for the features they implement. |
| **Use the application's inputs and measure on the target hardware.** | Samples from the user's dataloader expose realistic calls, while minimum and maximum shape checks verify that dynamic endpoints execute. Measurements give strategies evidence for selecting candidates against the chosen objective. | Broader coverage and more candidates cost tuning time. Changes in traffic, shape ranges, precision, or hardware can change the result. |
| **Support explicit tuning and tuning during inference.** | Teams preparing artifacts can control the tuning run. Teams integrating into an existing process can capture inputs through JIT and decide when to tune. Both use the module, strategy, and backend contracts. | Explicit tuning calls for setup and representative data. In-process tuning adds build work during application execution and depends on the calls the process observes. |

The application owns the request and its performance and quality targets. It also owns distributed setup, model
sharding, and the inference collectives that the model requires. AITune coordinates tuning within that environment
and evaluates module candidates compatible with it. See [Multi-GPU Integration](../multi_gpu.md) and
[Deployment Overview](../deployment/deployment.md).
