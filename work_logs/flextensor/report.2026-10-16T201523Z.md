<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlexTensor integration implementation report

Updated: 2026-10-16 20:02 UTC. Status: active; no completion claim.

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
| Integration adapter implemented | Pending | No production change verified yet |
| Real functional variants | Pending | Must run against the new implementation |
| Standalone AITune regression audit | In progress | No result yet |
| Real LTX mini benchmark | Feasibility in progress | No weights downloaded or run claimed |
| Source + outpainted videos | Missing | Required for completion |
| Lint/static/review | Pending | Required after implementation settles |

## Completion criteria

Completion requires checked source, passing meaningful tests on the actual implementation, unchanged core behavior, a real all-block LTX workload with measured performance/memory results, source/outpainted videos, and a final readable report. Missing or indirect evidence keeps the goal active.
