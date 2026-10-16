<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlexTensor integration implementation report

Updated: 2026-10-16 20:15 UTC. Status: adapter implemented; validation in progress; no completion claim.

## Scope

Fix and validate PR #89 without changing standalone AITune behavior or FlexTensor. Keep normal TorchInductor single-backend tuning, correctness checks, profiling, evaluation, and whole-block LTX offloading. Deliver a small real LTX benchmark and two short videos for manual inspection. See `plan.md` for the evolving checklist.

## Authoritative starting state

- Working repository: `/home/kkleczewski/repos/aitune`.
- Clean original main: `ed56e0902187556d5e07208ffbc74f31b08ce59c`.
- Verified remote heads: #43 `e9659c6ed3640f00056335e0d4b470048c7b00f6`; #45 `6afeb052c56631ae1bcdb363c33053a4ad609795`; #89 `6a63158b4bb5e540914ccae89e24f2ab24eeea4e`.
- Created local `kkleczewski/jit-flextensor` from the exact remote #89 head; no push or metadata changes.
- Earlier lazy-only fix `34d689ac920b9b1b89568319bdb87af5603e0f9d` remains preserved in `/tmp/aitune-jit-flextensor.COK8xd`; it is not the implementation base because it changes core configuration and narrows feature behavior.
- FlexTensor source reference: clean v0.5.0, commit `2561216341d1258d323fa84e5b0cd30e29765dd4`, `/tmp/flextensor-v0.5.0`.

## Findings carried forward (not substitutes for new acceptance tests)

1. The original real-GPU JIT021 eager/deferred failure was reproduced: isolated prepared-unit replay resubmits a rolling slot before its future consumer releases it, causing `TransferTicket ... is still in use` and partial eager fallback.
2. An integration-owned outer residency shell with a pure original compute child was GPU-tested against unchanged AITune at `6a63158`. Non-lazy Inductor compilation, built-in correctness checks, real profiling, graph detection, candidate fall-through, enabled performance gating, and replan were feasible. This prototype is not a production adapter or a full compatibility proof.
3. GitLab MR !449 (`fff0a135`) already used a provider revision containing the busy-slot check. It inherited mid-forward `join_after_forward()` calls absent in the current branch and supported a retained `compile_fn`/view-profile mode. Removing joins reproduces the failure; restoring them passes a small replay diagnostic but violates the provider's documented boundary and is not the selected fix.
4. MR !449 recorded all 48 blocks/96 Inductor routes and one-sample lower latency/device occupancy, but strict output checks failed. That does not establish the cause of numerical differences. The new LTX work will report diagnostics and provide playable clips for human review.
5. Previous JIT021 offload assertions proved nonzero rolling storage and less managed memory than all weights, not adherence to the requested fraction. A prior 352.086 MiB request produced 384.172 MiB managed residency. The new test must not repeat a false budget claim.

## Environment inventory

- NVIDIA RTX A6000, 49,140 MiB, driver 570.211.01; initially idle.
- Pinned image: `ghcr.io/ai-dynamo/aitune/nvcr-torch-26.01-py3@sha256:bd4b0ca692cec0d04dcfab89cb8cd1fc22a593e070c851308361d46a05fa67fb`.
- Known image runtime from prior checks: Python 3.12.3, Torch `2.10.0a0+a36e1d39eb.nv26.01.42222806`, CUDA 13.1 forward compatibility.
- Host RAM 31 GiB, swap 14 GiB, initially about 95 GiB free disk. LTX model cache contains configuration only. Host has `uv`; FFmpeg is not installed on the host.
- GitLab connector authentication was unavailable, but existing authenticated `glab` read access was verified in the earlier investigation; no credentials were changed. No secrets belong in logs or artifacts.

## Current work and ownership

- Integration adapter + focused unit tests: residency wrapper, live-target resolution, mapped replan invariants, retained-profile registration.
- Core invariance audit: public API/docs/tests, exact behavioral contract and regression commands; no core edits.
- LTX feasibility research: dependency/model/source-video plan and memory feasibility before downloading large weights.
- Main agent: plan/report, critical functional tests, serial GPU verification, documentation, final artifacts and integration review.

## Evidence ledger for this implementation

| Gate | Status | Evidence |
|---|---|---|
| Current stack heads and clean base | Verified | Local git status/branches and remote `ls-remote`, 2026-10-16 |
| Integration adapter implemented | CPU verified | 90 focused adapter tests passed; only `aitune/torch/jit/flextensor.py` production code changed |
| Real functional variants | Partly verified | Eager and deferred diagnostic/replan runs passed; enabled gate and retained-profile runs in progress |
| Standalone AITune regression audit | CPU verified | 668 passed, 79 explicit CUDA-required skips; source/API/config key invariance verified |
| Real LTX mini benchmark | Preparation in progress | Staged text checkpoint download active; no LTX generation or speedup claimed |
| Source + outpainted videos | Missing | Required for completion |
| Lint/static/review | Pending | Required after implementation settles |

## Completion criteria

Completion requires checked source, passing meaningful tests on the actual implementation, unchanged core behavior, a real all-block LTX workload with measured performance/memory results, source/outpainted videos, and a final readable report. Missing or indirect evidence keeps the goal active.

## First implementation results

The new public integration wrapper is `FlexTensorUnit(original_compute)`. It is inserted before provider preparation; only its final live `compute` child is registered through unchanged AITune. The adapter retains a small immutable boundary association with its registration, allowing replan to reject stale manager/model/shell/compute references without any global coordinator, provider-private state, rollback, or reset.

Focused adapter CPU suite: **90 passed**. Standalone CPU regression suite: **668 passed, 79 skipped** (all explicit unavailable-CUDA cases), 64 warnings, 293.73 seconds. Logs are under `tuning/ltx23_outpaint/logs/`, including `core-regressions-cpu.log`. Runtime registration signature, JIT config fields, default correctness/performance settings, and Inductor cache hashes were verified unchanged. `git diff --exit-code 6a63158 -- aitune ':!aitune/torch/jit/flextensor.py'` passed.

New real GPU results on the current adapter:

| Mode | Performance validation | Finalization | Selected routes | Correctness calls | Profiling measurements | Managed memory / strict cap |
|---|---|---|---:|---:|---:|---:|
| Deferred | Diagnostic (real profiling) | Done/applied replan | 10 Inductor | 10 | 600 | 335,724,544 / 369,188,864 bytes |
| Eager | Diagnostic (real profiling) | Done/applied replan | 10 Inductor | 10 | 600 | 335,724,544 / 369,188,864 bytes |

Both runs verified strict eager-output parity for shapes 1 and 2 and held-out values, real candidate builds, per-graph baseline/candidate throughput telemetry, one outer offload call per whole-model traversal despite repeated inner tuning calls, final live module identities, and preserved selected-backend dispatch after replan. These are integration correctness tests, not speedup claims: the tiny FP32 linear blocks were slower under Inductor, and diagnostic mode correctly records that without disabling profiling. A separate enabled-gate run checks ordinary measured eager selection.

The first strict budget check failed at 402,833,408 managed bytes versus the old 369,188,864-byte total request. We kept that total limit and separately reserved the excluded head's 67,125,248 bytes from the strategy fraction. The provider strategy request is now 302,063,616 bytes; actual managed residency is 335,724,544 bytes, below the unchanged total limit. This is explicit caller budget accounting, not a provider patch or a relaxed assertion. Compiler workspace and full-device peak are separate metrics.

One test-only issue was also caught: telemetry uses AITune report IDs, not Python `id(original_compute)`. The public report-ID lookup was corrected; no production behavior or numerical tolerance was altered.

## LTX preparation status

The new mini example is being added under `examples/LTX23OutpaintAITuneFlexTensor/`. It targets a real 17-frame clip at 24 FPS, 256x160 stage 1 and 512x320 final output, pinned pretrained weights and LoRA, all 48 transformer blocks, ordinary single-Inductor correctness/profiling, and separately staged prompt encoding. No quantized or randomly initialized proxy is permitted.

Memory feasibility is based on inspected source, not yet a native success: stored text-encoder shards total 48,749,441,672 bytes and can load as roughly 22.7 GiB BF16 directly on CUDA in a prompt-only process. Inference components total about 45.13 GiB on disk; the BF16 transformer can retain mmap-backed CPU storage during provider profiling. New task-owned prompt files are downloading under the ignored `tuning/ltx23_outpaint/models` cache. Only task-created temporary text weights may later be removed to make room for inference downloads; no existing user cache is selected for deletion.

The real public-domain Helen Hill source file (4,330,608 bytes) downloaded successfully after setting a normal descriptive HTTP User-Agent. PyAV 19's Theora decode currently fails; codec/environment diagnosis is in progress. No source clip or outpainted result is claimed until it is actually encoded and checked.
