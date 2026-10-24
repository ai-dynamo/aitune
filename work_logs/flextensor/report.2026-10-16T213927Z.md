<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlexTensor integration implementation report

Updated: 2026-10-16 21:33 UTC. Status: native matrix and complete tiny LTX comparison passed; larger short-workload experiment being prepared.

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
| Real functional variants | Passed, final telemetry persisted | JIT020 all four variants; final JIT021 four-case matrix and JIT022 native fallback passed in the healthy container |
| Standalone AITune regression audit | CPU verified | 668 passed, 79 explicit CUDA-required skips; source/API/config key invariance verified |
| Real LTX mini benchmark | Completed; gain marginal | Eager 11.614826 s, FT-only 21.725765 s, combined 21.208194 s; all 48 Inductor blocks; 29.3% allocated-memory saving versus eager |
| Source + outpainted videos | Generated; manual review pending | Actual source, eager/FT-only/combined and held-out clips exist; final playable links remain to be delivered |
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

## Final native matrix

All five fresh-interpreter final cases passed in `aitune-ft-dev-20261016b`. Public tuning JSON exists under `tuning/ltx23_outpaint/artifacts/<case>/`; matching logs are `logs/final-<case>.log`.

| Case | Actual result |
|---|---|
| eager-diagnostic-replan | 10 real Inductor routes; 10 correctness calls; 600 timing measurements; done/applied replan |
| deferred-diagnostic-replan | 10 real Inductor routes; 10 correctness calls; 600 timing measurements; done/applied replan |
| deferred-enabled-replan | All 10 candidates built/validated/profiled; all 10 measured eager baselines won this run; done/applied replan, no native fallback |
| deferred-diagnostic-profile | 10 real Inductor routes; 10 correctness calls; 600 timing measurements; retained profile, no replan request |
| native-fallback | Genuine fullgraph `Unsupported` from a compiler-disabled operation; original compute/hooks/outputs/residency preserved; zero replan calls |

Every case retained 335,724,544 reported managed bytes under the unchanged 369,188,864-byte total limit. Positive cases checked known/held-out outputs, live selected dispatch and one outer residency boundary per whole-model traversal. The enabled winner count differs from an earlier run (one Inductor/nine eager), as expected from actual measured latency selection; it is not a fixed expected count or a compilation failure.

## Host-pinning policy correction

The initial copied-buffer (`torch`) LTX offload attempt did not complete its benchmark. After its full profile was saved, host usage approached 29/31 GiB with about 21 GiB shared and active swap, while the GPU was mostly idle. It was deliberately interrupted with SIGINT; authoritative exit was 130, and the traceback was in Accelerate's connector `.to(execution_device)` transfer. The exact process was verified before interruption and then confirmed absent; CUDA occupancy returned to 1 MiB and host available memory to about 25 GiB. Its log is preserved as `logs/ltx-baseline-torch-pinning-interrupted.2026-10-16T205747Z.log`; no passing latency result is claimed.

FlexTensor 0.5.0 publicly supports `pinned_memory_mode="host_register"`, which registers existing allocations in place instead of creating copied pinned buffers. This is a supported caller policy, not a provider patch, omitted weights, reduced block coverage or disabled checks. The example now exposes that choice, checks public `flextensor.host_pinning.is_available()` before loading the large model, and records the requested mode. Actual allocation errors propagate. A capability check alone does not prove all large pins succeeded; inspect the completed native run and fallback warnings.

Both FT retries will explicitly reuse the same completed profile and `host_register` mode. The source profile was calibrated during the earlier copied-buffer attempt; retaining it is an intentional same-strategy comparison, not a claim of new host-register calibration. All LTX arms use the same recorded `TORCHINDUCTOR_COMPILE_THREADS=1`, three warmups and ten measurements. The original three-sample eager JSON was preserved as `eager/results.first-smoke.2026-10-16T2039Z.json` before the matched rerun.

## Matched real references and active combined run

| Measurement (three warmups, ten runs) | Resident eager | FlexTensor-only |
|---|---:|---:|
| Mean generation seconds | 11.6148258705 | 21.7257649820 |
| Peak allocated bytes | 47,331,982,848 | 33,443,462,656 |
| Peak reserved bytes | 48,345,645,056 | 34,496,053,248 |
| Sampled whole-device peak bytes | 49,318,395,904 | 35,517,038,592 |
| Sampled process RSS peak bytes | 12,509,499,392 | 26,454,343,680 |

Both arms completed structurally valid measured/held-out clips and finite held-out tensors. Public FT memory totals were 762,576,896 rolling-block bytes plus 24,634,415,616 unmapped bytes, total 25,396,992,512, with all 48 ordered whole-block units. Actual-offload and exact workload/package/compiler/measurement provenance checks passed. Held-out final video latents and audio waveform were **bit-identical** between resident eager and FT-only (maximum difference zero). This isolates any later compiled arithmetic differences from the provider-only path. The FT-only arm saves GPU memory but is slower than resident eager; do not hide that tradeoff.

The combined worker uses the same unchanged profile, host-register allocation policy and measurement settings. Public telemetry has 48 modules, 48 graph routes, 48 successful real Inductor candidate builds/selections, normal correctness enabled and nonzero eager/candidate throughputs. Example per-block measured speedups were about 1.43-1.52x; these are **not** end-to-end acceleration claims. Setup completed in 342.874 seconds including recording, tuning and route warmup. Final timed generations and held-out checks remain pending at this timestamp.

The first real combined run used the image's default JIT build-cache path `/tmp/integration-home/.cache/aitune.jit/a942f0dcc0`; its files were copied to `combined/jit-build-cache/` before the disposable container exits. The example now explicitly configures its public JIT cache path beneath the task directory for future runs; no core cache key/default was changed. The low-disk-space warning (47.95 GiB free versus the normal 50 GiB warning threshold) was retained rather than disabled; no disk-space failure has occurred.

## Container access mitigation and visual-review limits

Immediately after the successful matched eager worker, the long-lived `aitune-ft-dev-20261016b` container again lost NVML/CUDA access while the host remained healthy/idle. The baseline attempt failed before model loading; its log is `logs/ltx-baseline-container-device-failure.2026-10-16T2105Z.log`. Its packages were preserved in local-only image `aitune-flextensor-dev:20261016b`, image SHA256 `7c4e74111b4383d0674467de8979071f5f1e0dd5875d78762a0b598013db8145`. No image was pushed.

Docker uses the systemd cgroup driver. The symptoms match NVIDIA's documented container device-access-loss class, but the triggering external event was not identified. Fresh per-arm containers now request the same GPU plus its explicit existing `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm` and `/dev/nvidia-uvm-tools` nodes with `--runtime=nvidia`; both use the original unlimited memlock/64 MiB stack settings. This follows a [per-container mitigation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html#containers-losing-access-to-gpus-with-error-failed-to-initialize-nvml-unknown-error) without editing/restarting the host daemon or driver. The completed FT reference and active combined worker retain healthy device access.

Actual source/eager frames were inspected. The resident-eager output already changes source expression/appearance at this small resolution. Pinned native Diffusers conditioning appends clean reference tokens; it does not hard-copy the original center. Black padding is exact RGB zero and matches the [pinned LoRA model card](https://huggingface.co/oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint/blob/b747e10700e7460f90d0cb1ceb061f69c719431a/README.md). No mask corruption was found. Keeping outpaint LoRA active but omitting stage-2 references matches the historical MR's explicit native workload choice, distinct from FT's separately patched both-stage recipe. Its effect on perceptual preservation is not proven; do not attribute visible changes to integration or call perceptual acceptance passed.

## Completed tiny comparison and next experiment

All three matched modes and `compare` completed successfully. Raw JSON, latencies, videos and held-out tensors are under `tuning/ltx23_outpaint/`; the authoritative comparison is `comparison.json` and log `logs/ltx-compare.log`.

| Three warmups / ten runs | Resident eager | FT-only | AITune + FT |
|---|---:|---:|---:|
| Mean generation seconds | 11.6148258705 | 21.7257649820 | 21.2081939233 |
| Peak allocated bytes | 47,331,982,848 | 33,443,462,656 | 33,443,462,656 |
| Peak reserved bytes | 48,345,645,056 | 34,496,053,248 | 34,623,979,520 |
| Sampled whole-device bytes | 49,318,395,904 | 35,517,038,592 | 35,647,062,016 |
| Sampled process RSS bytes | 12,509,499,392 | 26,454,343,680 | 26,984,796,160 |

The combined mean is about 1.0244x the FT-only throughput, a marginal 2.4% latency reduction within observed run drift. It is not a convincing acceleration claim. Both FT modes save 29.3% peak allocated GPU memory relative to resident eager, which is faster on this tiny workload. Faster individual compiled blocks do not establish an end-to-end gain. Transfer/non-transformer overhead plausibly dominates here; that is an inference, not an isolated timing proof.

The profile digest remained `784428c8f940be44b3663b2eaf763a2e74b05ce6be15273444309e2e1c61c787`. All 48 combined blocks selected real Inductor routes and recorded positive eager/candidate throughput. FT-only held-out latents/audio are bit-identical to eager. Combined video-latent maximum absolute difference is 0.1515541, relative L2 0.0180486 and cosine 0.9998384; audio maximum difference is 0.0959473, relative L2 0.3177713 and cosine 0.9486070. Both fail the explicit 1e-2 allclose diagnostic; all values are finite. No numerical equivalence or perceptual acceptance is claimed.

The actual 17-frame source and 512x320 combined clips exist. To investigate whether more useful short-workload compute changes the tradeoff, one additional geometry is being prepared in `tuning/ltx23_outpaint_compute/`: stage 1 768x480, final 1536x960, still 17 frames and all 48 unchanged whole blocks. Only caller geometry changes. Reuse model files and the resolution-independent prompt; generate a new source and a new FT profile. The completed tiny result stays intact. No gain is assumed before this experiment runs.
