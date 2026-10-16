<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlexTensor integration implementation report

Updated: 2026-10-16 20:55 UTC. Status: native cases and first real LTX eager run passed; real FT profile saved; combined validation remains pending.

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

## Historical GitLab comparison

Read-only GitLab API verification confirmed the current MR heads: !343 `d32ca8496874730a13bf1d709d99d1dc10a9b217`, !359 `4d440c566d9e2687c79f87457c52e54ebfe4314e`, !448 `11395c1e42bcd1c10aea2169c762036b60c5cbf5`, and !449 `fff0a1352e5daa9be93fa9da8df1909530598695`. These are different experiments, not interchangeable evidence.

| MR | What was established | What was missing |
|---|---|---|
| !343 | 49 frames, 1280x704, five warmups/50 measurements; 20.488 to 17.768 seconds; 45.790 GiB peak allocated; all 48 blocks/192 Inductor routes; compiled-timing replan | Strict eager-versus-compiled output-value validation was explicitly absent |
| !359 | Fresh-worker benchmark workflow | Explicitly no eager output-value comparison |
| !448 | Pure FlexTensor completed 45 seconds; a separate 49-frame combined smoke selected eager for one resident `proj_in` target and was bit-exact but 15.3% slower | No 45-second all-block combined benchmark |
| !449 | Actual 45.042-second, 1081-frame, 1920x1088 run; all 48 blocks/96 Inductor routes; retained profile; 6141.440 to 4530.608 seconds and 95.579 to 65.309 GiB sampled NVML peak | Zero warmups/one measured generation; strict short/full-output video and audio gates failed; perceptual quality not established by those numeric gates |

Evidence lives in !343 `examples/LTX23OutpaintAITuneFlexTensor/RESULTS.md:10,35,57`; !359's README at line 151; !448's README at lines 338-365; and !449's README at lines 379-445. !449 reports video maximum difference 77 uint8 levels and audio maximum difference 0.552856; its own README labels 1.355544x a raw latency ratio, not correctness-accepted speedup. These differences do not establish their cause or prove that a human would reject the video.

Two concrete changes in the current stack matter independently:

1. **Transfer lifecycle regression:** !449 `aitune/torch/jit/flextensor.py:692-721` joined transfers before/after tuning and after individual backend calls, connected through the backend inference wrapper. Current base `6a63158` still registers provider-prepared units but removes those joins, so normal AITune warmup/check/profile replay also re-enters provider scheduling during a paused whole-model traversal. Real reproduction confirms the busy-slot failure. Restoring those joins is not the selected fix: FlexTensor v0.5.0 `src/flextensor/loaders.py:1622-1631` explicitly says the join is for end-of-forward/piece, not mid-forward. The new outer-shell/original-compute boundary avoids isolated transfer replay while preserving ordinary core calls.
2. **Retained-profile capability removed:** current base adapter lines 33-34 require external compilation and reject `compile_fn`; !449 adapter lines 46-77 and README line 197 supported the retained view-profile path. That is a separate integration-scope reduction. The new adapter accepts both explicit replan and retained-profile workflows without equating them.

!449 pinned provider `b94be2f813f18f347f222583ca3b536c2bfef20a`, which already contained the busy-slot guard. Consequently the evidence does not support blaming the guard's appearance on the 0.5.0 release. No numerical discrepancy is attributed to offload ownership without a separate controlled test.

The published !449 command passed a shorter `--aitune-setup-video`: `benchmark.py:1453-1457` calls `configure_aitune(inductor_dynamic=True)`, whose default `correctness_only=False` selects real eager/candidate profiling and enabled `MinLatencyStrategy` evaluation (`outpaint.py:359-367`). Reporting explicitly records performance selection enabled on those setup-video shapes. The helper contains an unused profiling-disabled branch; that branch is not evidence that the published experiment skipped profiling. The long target was separately executed and numerically compared.

## Environment inventory

- NVIDIA RTX A6000, 49,140 MiB, driver 570.211.01; initially idle.
- Pinned image: `ghcr.io/ai-dynamo/aitune/nvcr-torch-26.01-py3@sha256:bd4b0ca692cec0d04dcfab89cb8cd1fc22a593e070c851308361d46a05fa67fb`.
- Known image runtime from prior checks: Python 3.12.3, Torch `2.10.0a0+a36e1d39eb.nv26.01.42222806`, CUDA 13.1 forward compatibility.
- Host RAM 31 GiB, swap 14 GiB, initially about 95 GiB free disk. Task-local inference weights and encoded prompt are now cached. Host has `uv`; PyAV handles video without a system FFmpeg executable.
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
| Real functional variants | Passed; final artifact reruns pending | JIT020 all four variants; JIT021 eager/deferred diagnostic replan, deferred enabled gate and retained-profile; JIT022 genuine unsupported-fullgraph native fallback |
| Standalone AITune regression audit | CPU verified | 668 passed, 79 explicit CUDA-required skips; source/API/config key invariance verified |
| Real LTX mini benchmark | Eager reference passed | First resident eager mean 13.052329 seconds; offload baseline running; combined not yet run |
| Source + outpainted videos | Partial | Actual source and eager-reference clips exist; requested combined clip still missing |
| Lint/static/review | Passed so far | Changed Python Ruff/format, whitespace, focused adapter pytype and repository pre-commit passed; adapter/docs reviews found no actionable issues; final sweep remains after runtime fixes |

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

The real public-domain Helen Hill source file (4,330,608 bytes) downloaded successfully after setting a normal descriptive HTTP User-Agent. PyAV's Theora decoder emits empty repeat packets; skipping those packets and resampling decoded frames by timestamp resolved source preparation. The actual source MP4 contains 17 frames at 160x96/24 FPS. The centered first-stage reference canvas was visually inspected and its exact-zero border checked.

Prompt encoding succeeded in about 51.08 seconds. `prompt.pt` contains finite BF16 embeddings of shape `[1, 1024, 188160]`, with a `[1, 1024]` mask and pinned provenance. Only the eleven explicitly resolved, newly downloaded task-cache text-weight blobs were subsequently removed (about 45.4 GiB, publicly downloadable again). Existing user caches were not touched. Inference component downloads completed; the CPU mmap model load succeeded without host OOM.

The first real resident-eager run completed with all 48 pretrained blocks and the published LoRA: latencies 13.556587, 13.074842, and 12.525559 seconds, mean 13.052329 seconds. Measured peaks were 47,331,982,848 allocated bytes, 48,345,645,056 reserved bytes and 49,318,395,904 sampled whole-device bytes. This is a reference result, not an integration gain. Its JSON predates the latest shared output checks/PEFT provenance field, so it must be rerun before the final like-for-like comparison. Eager video artifacts exist; the requested combined output is not yet available.

An initial inference attempt stopped on a missing PEFT dependency before generation. Installing and pinning `peft==0.18.1` resolved it without modifying Torch, provider or core. The example now checks package provenance before loading large weights and verifies video structure plus finite held-out latents/audio in every mode, outside the timing interval.

## Additional native evidence and environment incident

The deferred enabled-performance case selected one Inductor route and nine legitimately measured eager routes; all ten targets became `TunedModule`, with no native build failure. The retained-profile case selected ten Inductor routes without requesting replan. The genuine negative test used a real compiler-disabled operation, reached fullgraph Inductor warmup failure, preserved native eager compute/hooks/held-out outputs and offloading, and requested no replan. All four JIT020 placement/mode combinations passed.

An attempted stronger assertion that `unmapped_tensors_bytes` equaled the excluded head was incorrect and removed: provider unmapped accounting includes all reachable CUDA tensors absent from the view mapping. Component sum, nonzero rolling blocks and the strict unchanged total cap remain asserted. Final per-case JSON reruns are still required after this test-only correction.

The original long-lived development container later lost CUDA/NVML device access while host GPU access and a fresh container probe remained healthy. No driver or application source change was needed: a replacement container, `aitune-ft-dev-20261016b`, uses the same installed runtime/workspace and has passed the real eager run. NGC forward compatibility requires `bash -lc` initialization for CUDA on this host. Failed final-case logs from the affected container are not passing evidence and will be replaced by actual reruns.

## Real offload baseline progress

The active baseline completed view profiling and saved `tuning/ltx23_outpaint/flextensor-profile/profile.json`. Its SHA256 before combined reuse is `784428c8f940be44b3663b2eaf763a2e74b05ce6be15273444309e2e1c61c787`. The saved parameter mapping includes every whole-block `compute` subtree and nested PEFT base/LoRA weights. Diagnostics planned about 13.65 GiB of offload from 35.79 GiB of block weights; this is a strategy report, not yet a completed steady-state benchmark or proof of a whole-device cap.

This setup is slow because the 35+ GiB checkpoint exceeds host RAM. Provider source confirms one discovery, three unmeasured view-profile compile warmups and ten measured root forwards. Each repacks its block from original file-backed storage; observed reads exceeded 500 GB. Source inspection found shared Parameter storage, not a second whole-model clone. Public profile existence now proves startup progressed beyond the point where an entire generation returned in inference; it does not by itself prove subsequent measured/held-out generations passed.

The example now prints non-timed startup milestones and performs comparison-side checks for matching workload, package provenance, finite compiler limits, warmups/runs, FT memory fraction, finalization and profiling settings. Retained-profile comparison requires explicit profile reuse from the same location. Both FT result records must prove positive rolling storage, consistent public memory components, managed residency below the unquantized BF16 weight lower bound, and all 48 ordered whole-block units. These are checked against actual result data rather than inferred from configuration.

Repository pre-commit hooks passed for the changed files. Functional metadata validation passed for 92 scripts and 14 projects with zero errors; its existing multiple-metadata warning for JIT001 is unrelated. Independent read-only adapter and documentation reviews found no actionable mismatches. No commit or push has been made.
