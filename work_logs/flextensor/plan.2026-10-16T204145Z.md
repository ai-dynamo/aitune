<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlexTensor integration improvement plan

Updated: 2026-10-16 20:15 UTC. Status: adapter implemented; GPU and LTX validation in progress.

## Objective and invariants

Improve PR #89 on top of #43 (explicit registration) and #45 (external placement), using unmodified FlexTensor 0.5.0. Standalone AITune must retain its existing API, defaults, configuration/cache identities, per-module compilation, correctness checks, profiling, performance evaluation, and fallback behavior. Do not introduce reset/recovery/session machinery or a lazy-only backend mode.

The initial integration backend is TorchInductor JIT. Preserve ordinary single-backend tuning and validation, and leave a straightforward extension point for later backends. Offload and tune whole LTX 2.3 transformer blocks, not a reduced proxy model.

## Implementation direction

Use one integration-owned outer `FlexTensorUnit` per selected block before FlexTensor preparation. FlexTensor manages the outer unit; its `compute` child is the original module, registered through unchanged AITune explicit external-placement registration after FlexTensor reaches inference. Repeated AITune build/check/profile calls execute only the resident compute child, inside a single provider enter/exit boundary. Resolve final live units through public APIs; never manipulate provider-private tickets, weights, or placement.

Keep the existing explicit full-model replan helper. Map registered compute children back to current provider shells and verify identities and dispatch after replanning. Also preserve the separate public compiled-profile/retained-strategy workflow, without silently treating it as a successful replan.

## Work sequence

1. [x] Verify working tree and remote stack heads; preserve the earlier unpushed lazy-only experiment separately.
2. [x] Read repository guidance and tuning/benchmark/validation skills; review prior MR findings and the shell feasibility prototype.
3. [x] Audit public AITune behavior, docs, and regression tests; record invariants and corner cases.
4. [x] Implement the residency boundary only in the integration adapter and focused contract tests.
5. [ ] Strengthen real GPU functional coverage: eager/deferred, both known shapes and held-out values, normal correctness, real profiling, no partial fallback, actual offloading, and preserved live compiled dispatch after replan.
6. [x] Check wrapper signatures, hooks, structured outputs, selection order, alias/nesting rejection, invalid provider state, and retained-profile mode in focused contract tests (real provider matrix still in progress).
7. [ ] Run JIT020's four variants, relevant ordinary AITune suites, lint/static checks, and a source-diff audit proving no core behavior/config changes.
8. [ ] Establish a feasible real pretrained LTX 2.3 mini workload on the A6000: short clip and low resolution, all 48 transformer blocks, fixed weights/settings, caller-owned compiler limits, and reproducible dependencies.
9. [ ] Run eager/offload baselines before tuning; measure setup, repeated inference latency, total and managed memory separately; run the combined workload and record numerical diagnostics without substituting them for human review.
10. [ ] Produce two short playable videos (source and combined outpaint), plus complete readable results and remaining limitations.
11. [ ] Review all requirements against actual evidence before completion. Keep the goal active while any requested deliverable is missing.

## Validation rules

- Functional tests must detect actual feature failures, not mask them with test-side provider-state changes, fake compilation, reset, omitted targets, permanent resident weight copies, or weakened tolerances.
- Ordinary AITune correctness checks remain enabled. Strict toy-model comparisons remain strict. LTX numerical differences are reported as diagnostics alongside finite/structure checks and final manual visual review; do not claim numerical equivalence that was not demonstrated.
- Real profiling must be observed, not inferred from configuration. A performance gate may legitimately select eager; distinguish that from compilation failure/native fallback.
- Budget accounting distinguishes managed weights/rolling blocks from activations, compiler allocations, allocator reserve, and whole-device occupancy. Do not claim a cap from an assertion that only proves some offload.
- Each functional variant runs in a fresh interpreter. Normal workflow registers once and uses complete model traversals, without reset or rollback.
- Do not push or alter PR metadata. Revisit local commit/signing requirements before committing.

## Current constraints and risks

- GPU: RTX A6000, 49,140 MiB. Host RAM: 31 GiB, swap: 14 GiB. Free disk initially about 95 GiB. Only LTX configuration is cached, not model weights. Lower resolution reduces activations, not the 22B model weight footprint; establish a feasible loading/offload plan before a large download.
- Wrapper insertion adds a `compute` component to module/state-dict paths. This is an explicit integration topology choice; preserve original module objects/hooks and document serialization limitations instead of inventing key-remapping machinery.
- Prior MR !449 used the same busy-slot guard but an unsupported mid-forward join workaround and a retained-profile path. Its reported speed/memory improvement was not strict numerical acceptance. Neither that workaround nor a lazy-only feature reduction should be ported as the fix.
- The strengthened native test initially exposed the known provider budget discrepancy before tuning. Preserve the strict half-weight total residency cap (369,188,864 bytes), but explicitly reserve the excluded resident head (67,125,248 bytes) from the provider strategy fraction. The actual new native replan result is 335,724,544 managed bytes, below that unchanged cap. Record both strategy request and actual total rather than claiming that a provider fraction alone enforces a physical cap.
- LTX feasibility analysis found an unchanged-weight path: stage prompt encoding separately on CUDA, retain BF16 transformer safetensors as CPU mmap during profiling, and keep sufficient weights GPU-resident that provider-pinned host tensors fit RAM. The prompt checkpoint download is active. Verify memory experimentally before claiming this path works.

## Decisions from the first iteration

- Adapter CPU contract tests: 90 passed. Standalone core regression suite: 668 passed, 79 explicit CUDA-required skips. Core source/API/configuration hashes remain unchanged.
- First new real-provider deferred and eager variants passed all ten Inductor targets, ten built-in correctness checks, 600 actual profiling measurements, known and held-out outputs, strict managed-memory cap, and completed/applied replan with retained dispatch.
- The functional test's first telemetry check used Python object IDs incorrectly. Corrected it to join public registration report IDs to public tuning-data IDs; no production behavior changed to satisfy that assertion.
- LTX input download succeeded after normal HTTP User-Agent handling. Its Theora decode currently needs an environment/codec fix; do not substitute a fake model or fabricated source/output.

## History

Timestamped copies of this file and `report.md` live in `work_logs/flextensor/`. Archive before material plan/status revisions, and record why the next action changed.
