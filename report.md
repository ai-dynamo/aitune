<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlexTensor integration implementation report

Updated: 2026-10-16 22:28:25 UTC. Status: implementation and validation complete; packaged as one intentionally unsigned local DCO commit. No signing or push.

## Current outcome

The integration fix isolates FlexTensor's residency shell from the original compute child registered with AITune. Only `aitune/torch/jit/flextensor.py` changes production code; standalone AITune and FlexTensor remain unchanged.

Both required eager/deferred native cases select all ten real Inductor targets, preserve strict known/held-out outputs, satisfy the independent managed-memory cap, complete done/applied replanning and retain selected dispatch. Retained-profile, normal enabled performance selection and genuine native compiler-failure fallback are tested separately.

The larger real all-48-block LTX comparison completed: AITune + FT mean latency is 23.944830 s versus matched FT-only 26.699703 s (10.3% lower latency) and resident eager 25.578774 s (6.4% lower). Peak allocated GPU memory is 28.3% below resident eager. This is a promising single-machine result, not a statistical/general-workload guarantee or perceptual-quality acceptance. The smaller default geometry remains preserved: its 2.4% mean improvement over FT-only is marginal, within observed drift, and resident eager is faster.

| Gate | Actual status |
|---|---|
| Adapter contracts | 90 CPU tests passed |
| Standalone regression | 668 passed; 79 explicit CUDA-required skips |
| Prepared units / native provider | JIT020 four variants and final five-case JIT021/JIT022 matrix passed |
| Tiny real LTX | Eager, FT-only, combined and comparison all exited successfully |
| Larger short workload | Eager 25.578774 s; matched reuse FT-only 26.699703 s; combined 23.944830 s / all 48 Inductor; final comparison passed |
| Static checks | Final pre-commit passed across task files; metadata 92 scripts / 14 projects / zero errors; whitespace/source invariance passed |
| Handoff | Playable videos and runnable command manifest below; one local Conventional/DCO commit, intentionally unsigned and unpushed |

## Starting state and ownership

Repository: `/home/kkleczewski/repos/aitune`; branch: `kkleczewski/jit-flextensor`.
Verified starting heads: #43 `e9659c6ed3640f00056335e0d4b470048c7b00f6`, #45 `6afeb052c56631ae1bcdb363c33053a4ad609795`, #89 `6a63158b4bb5e540914ccae89e24f2ab24eeea4e`.
The clean original main was `ed56e0902187556d5e07208ffbc74f31b08ce59c`.

The previous unpushed lazy-only fix `34d689ac920b9b1b89568319bdb87af5603e0f9d` remains in `/tmp/aitune-jit-flextensor.COK8xd`; it is not this implementation's base because it changes core configuration/behavior.
Provider reference: unmodified FlexTensor v0.5.0, commit `2561216341d1258d323fa84e5b0cd30e29765dd4`, under `/tmp/flextensor-v0.5.0`.
No provider-private tickets/placement, shared-core changes, reset/recovery, signing configuration, remote history or PR metadata were modified.

## Root cause and retained experimental evidence

At the tested base, AITune registers provider-prepared offload units. Inductor compilation/warmup and ordinary correctness/profiling replay each unit while its enclosing model traversal is paused. Those replays also resubmit FlexTensor transfers, colliding with a rolling slot whose future consumer has not released it.

Supplemental fresh-interpreter runs of the isolated exact `6a63158` checkout both exited 1 with `{'tuned': 4, 'eager': 6}`. Persistent first-cause backend logs distinguish the current GPU's scheduling from the original CI trace:

| Mode | First preserved cause | Evidence below `tuning/ltx23_outpaint/` |
|---|---|---|
| Eager | `schedule=layers.1 consume=layers.3`, ticket still in use | `artifacts/supplemental-base-eager-cache/41c1a4b68f/0/TorchInductorJitBackend_1adfb91fedaeacea61744f373e9f16fff6fd80be937d174b47e30d3fdfa7865e/build.log:5` |
| Deferred | `schedule=layers.2 consume=layers.3`, ticket still in use | `artifacts/supplemental-base-deferred-cache/d8f788cc5d/0/TorchInductorJitBackend_1adfb91fedaeacea61744f373e9f16fff6fd80be937d174b47e30d3fdfa7865e/build.log:5` |

Matching outer logs are `logs/supplemental-base-{eager,deferred}-persistent.log`.
The provider-supported lazy harness exited 0: ten prepared/lazy-compiled units, twenty new Inductor graphs, eight complete root forwards, both known shapes plus held-out values and zero isolated replays.
Its source and output are `artifacts/supplemental-provider-lazy.py` and `logs/supplemental-provider-lazy.log`; it follows provider `tests/integration/_compile_helpers.py:240-244`.

That diagnostic had 134,217,728 rolling-block bytes and 402,833,408 managed bytes versus 738,377,728 total weight bytes. It proves real offloading, not adherence to the later strict half-weight cap.
These are supplemental tests on the current runtime, not retroactive proof that dependencies had never changed before the first investigation.
Earlier tool-observed reproductions/prototypes informed the fix; the supplemental files are the retained raw evidence. The two old `/tmp/aitune_job_*.log` downloads are HTTP 403 responses, not CI logs.

## Minimal implementation and invariants

Insert `FlexTensorUnit(original_compute)` before provider preparation and select the whole shell. After inference starts, resolve final live shells from public APIs and register their original `compute` children through unchanged `register_for_jit_tuning(..., device_management="external")`.
Repeated AITune calls now compute while the enclosing shell keeps weights resident; they do not replay an outer transfer boundary.

A small registration association retains manager/model/shell/compute and prepared-forward identities. Explicit synchronous replanning maps the selected children back to live public shells, invokes exactly the requested number of complete model forwards and verifies done/applied status plus preserved dispatch.
The separate identity-`compile_fn` retained-view-profile route keeps its pre-AITune strategy and does not request or claim a replan. Arbitrary `compile_fn` shell transformations are not established.

Original module/parameter objects, hooks, argument forwarding and outputs are preserved. Wrapping intentionally adds `compute.` to state-dict paths; old unwrapped checkpoints are not automatically interchangeable. The shell is not general attribute/type transparency.
Normal warmup, correctness, graph-break detection, per-module strategy cloning, profiling, evaluation and fallback stay in their existing core paths. Ordinary external-placement candidate restrictions still apply.

Core source invariance passed:
`git diff --exit-code 6a63158 -- aitune ':!aitune/torch/jit/flextensor.py'`.
Public registration/configuration fields and runtime Inductor cache keys were unchanged; there is no lazy-compilation config addition.
Default/configured-fullgraph Inductor hashes remain `968546feca89006f602383244f9b3d9489aab783d76efaab7ff61532ebacfeb7` and `1adfb91fedaeacea61744f373e9f16fff6fd80be937d174b47e30d3fdfa7865e`.

## Validation on the current adapter

Focused adapter tests: 90 passed. Standalone CPU suites: 668 passed, 79 explicit CUDA-required skips, 64 warnings, 293.73 seconds; see `logs/core-regressions-cpu.log`.
Coverage includes JIT, module/signature/recording/dispatch, strategies, correctness, profiling, checkpoint storage and selected backend tests. Skipped GPU tests are not counted as passes.
JIT020 eager/deferred with both AITune/external placement passed; logs are `logs/jit020-<mode>-<placement>.log`.

All five final provider variants passed in fresh interpreters. Logs are `logs/final-<case>.log`; public JSON is under `artifacts/<case>/`.

| Case | Actual selection / result |
|---|---|
| eager-diagnostic-replan | 10 Inductor; 10 correctness calls; 600 profiling measurements; done/applied |
| deferred-diagnostic-replan | 10 Inductor; 10 correctness calls; 600 profiling measurements; done/applied |
| deferred-enabled-replan | All 10 Inductor candidates built/validated/profiled; measured eager baselines won all 10 routes; done/applied |
| deferred-diagnostic-profile | 10 Inductor; 10 correctness calls; 600 profiling measurements; retained strategy, no replan request |
| native-fallback | Genuine fullgraph `Unsupported`; original eager compute/hooks/outputs/residency preserved; zero replan calls |

The enabled winner count differed from an earlier run (one Inductor/nine eager), as expected for a measured gate; a tuned eager selection is not native build-failure fallback.
Positive cases checked both shapes and held-out values, real build/profiling telemetry, one outer residency call per full traversal, original live identities and selected dispatch after finalization.

All final cases report 335,724,544 managed bytes under an unchanged 369,188,864-byte total limit.
An initial strict check caught 402,833,408 bytes. Reserving the excluded 67,125,248-byte head from the caller's strategy fraction gives a 302,063,616-byte strategy request; the total cap was not relaxed.
Managed components are checked, but unmapped bytes do not equal head bytes: provider counting covers every reachable CUDA tensor absent from its view mapping.
Compiler workspace, activations, allocator reserve, whole-device occupancy and transient peaks are separate metrics. Telemetry joins use public registration report IDs, not Python object IDs.

Repository pre-commit passed, as did changed-Python Ruff/format, whitespace and focused adapter pytype checks. Final handoff logs are `logs/handoff-precommit.log` and `logs/handoff-functional-metadata.log`.
Functional metadata validation passed 92 scripts and 14 projects with zero errors; JIT001's pre-existing multiple-metadata warning is unrelated. Independent adapter/docs reviews found no actionable findings.
The handoff sweep includes all task files/history and is repeated after the final status-only report edits. No full repository-wide pytype or all-GPU functional-suite pass is claimed.

## Real tiny LTX workload and measurement

The example uses a real 17-frame / 24 FPS Helen Hill clip: 160x96 source, 256x160 stage 1 and 512x320 final output.
All 48 whole pretrained blocks and the pinned published outpaint LoRA remain active; no quantization, random proxy or reduced target set is used.
Both stages retain their published distilled schedules. Stage 1 has the zero-padded reference; stage 2 keeps the LoRA without another reference condition. This is a declared workload choice, not equivalence to the historical 45-second experiment.

Each arm runs in a fresh process with three warmups and ten timed full two-stage generations; output checks/encoding are outside the timed interval.
Weights, source/prompt hashes, conditioning, package/compiler settings and FT policy match. All arms record `TORCHINDUCTOR_COMPILE_THREADS=1`.
Both FT arms use `host_register`, fraction 0.65 and the same explicitly reused retained profile:
`784428c8f940be44b3663b2eaf763a2e74b05ce6be15273444309e2e1c61c787`.
Its digest remained unchanged. It originated in the earlier copied-buffer calibration; this is a same-strategy comparison, not a new host-register calibration.

| Three warmups / ten runs | Resident eager | FT-only | AITune + FT |
|---|---:|---:|---:|
| Mean generation seconds | 11.6148258705 | 21.7257649820 | 21.2081939233 |
| Peak allocated GPU bytes | 47,331,982,848 | 33,443,462,656 | 33,443,462,656 |
| Peak reserved GPU bytes | 48,345,645,056 | 34,496,053,248 | 34,623,979,520 |
| Sampled whole-device peak bytes | 49,318,395,904 | 35,517,038,592 | 35,647,062,016 |
| Sampled process RSS peak bytes | 12,509,499,392 | 26,454,343,680 | 26,984,796,160 |
| Setup seconds | 46.8321 | 72.5296 | 342.8736 |

Combined improves mean latency by only about 2.4% versus FT-only (1.0244x throughput), within observed run drift. This is not convincing end-to-end acceleration.
Resident eager remains faster; both offloaded arms save 13,888,520,192 allocated bytes, or 29.3%. Individual compiled blocks being faster does not establish pipeline acceleration.
Transfer/non-transformer overhead plausibly dominates this tiny case; that is an inference, not isolated attribution.

Combined telemetry contains 48 modules/graphs and 48 successful real Inductor selections, with ordinary correctness enabled and positive eager/candidate throughputs for every graph.
Both FT results list all 48 ordered shell units and report 762,576,896 rolling-block plus 24,634,415,616 unmapped bytes: 25,396,992,512 managed bytes.
Positive rolling storage and managed residency below the unquantized weight lower bound establish actual offload; the configured fraction is not a hard whole-device cap.

Authoritative artifacts are `tuning/ltx23_outpaint/{eager,baseline,combined}/results.json`, `comparison.json` and `logs/ltx-compare.log`; all three arms and comparison exited successfully.
Build evidence from the initial default cache was preserved in `combined/jit-build-cache/`. Future runs explicitly set the public JIT cache beneath the work directory; no core key/default changed.

## Tiny numerical diagnostics and human review

FT-only held-out final video latents and audio are bit-identical to resident eager, maximum difference zero.
Combined versus either reference is finite but not numerically equivalent:

| Held-out tensor | Maximum absolute difference | Relative L2 | Cosine |
|---|---:|---:|---:|
| Video latent | 0.1515541 | 0.0180486 | 0.9998384 |
| Audio waveform | 0.0959473 | 0.3177713 | 0.9486070 |

Both fail the explicit atol=rtol=1e-2 allclose diagnostic. These diagnostics neither relax built-in checks nor act as an automatic perceptual gate.
The unchanged built-in checks assess finiteness, shape and mutation/dynamic-boundary contracts; they are not eager-value allclose.

Actual source, eager and combined frames were inspected. Eager and combined look very similar, but both change source expression/appearance.
Native conditioning appends reference tokens rather than hard-copying the center; the exact-zero padding was checked. No mask corruption was established.
No human quality acceptance, source-identity preservation or long-video equivalence is claimed.

Playable tiny artifacts exist at `source.mp4`, `combined/outpaint.mp4` and `combined/heldout_outpaint.mp4` beneath the tiny work directory, alongside eager/FT-only references and held-out tensor files.
Final user-facing playable links remain part of handoff.

## Runtime provenance, memory constraints and resolved incidents

Hardware: NVIDIA RTX A6000, 49,140 MiB, driver 570.211.01; host RAM 31 GiB and swap 14 GiB.
Base image: `ghcr.io/ai-dynamo/aitune/nvcr-torch-26.01-py3@sha256:bd4b0ca692cec0d04dcfab89cb8cd1fc22a593e070c851308361d46a05fa67fb`.
Runtime: Python 3.12.3, Torch 2.10 NGC26.01 (`a36e1d39eb`), CUDA 13.1 forward compatibility, FlexTensor 0.5.0, Transformers 4.57.6, Accelerate 1.15.0, PyAV 19.0.1 and PEFT 0.18.1. Per-run JSON retains exact package versions.
The local-only derivative image is `aitune-flextensor-dev:20261016b`, SHA256 `7c4e74111b4383d0674467de8979071f5f1e0dd5875d78762a0b598013db8145`; no image was pushed.

Diffusers 0.39.0.dev0 was installed from the [ZIP of commit `9159a587b1f148a2cc911deb72b6d755508ef38b`](https://github.com/huggingface/diffusers/archive/9159a587b1f148a2cc911deb72b6d755508ef38b.zip).
Its full `direct_url.json` records archive SHA256 `a4de10c581072337bf25a1641b6876081e43bba4bcc00f1567004f89725e7178`.
Read-only comparison with primary GitHub raw source returned HTTP 200 and byte equality for `pipeline_ltx2_ic_lora.py` (117807 bytes; SHA256 `6cc7eb9aa29650c64ca287d8babcaaf9cee93964de792b246c466abb6873f7f1`) and `transformer_ltx2.py` (77968 bytes; `a6f5a201b670c2a5fa5d1ca1f5dfbd87b6330541cf5a9f0fa7605ea54cb3ce4c`).
This verifies those files, not every installed file or an independently downloaded ZIP. Old result JSONs retain null VCS revision because archive installs lack `vcs_info`; future runs record archive-source metadata explicitly.

Prompt encoding succeeded separately in about 51.08 seconds with finite BF16 embeddings `[1,1024,188160]`; the mask is `[1,1024]`.
Only eleven resolved task-owned text-weight blobs were removed afterward, freeing about 45.4 GiB; existing user caches were not touched.
A missing PEFT dependency was resolved before successful generation. The first three-sample eager smoke is preserved separately and is not used in the matched table.

View startup requires one discovery, three unmeasured warmups and ten measured profile root forwards by default; it repeatedly touches mmap-backed weights larger than host RAM.
Source inspection shows shared Parameter storage plus one largest-label/shared-prefix staging block, not a mandatory whole-model CPU clone. More than 500 GB of reads was compatible with finite progress.
The copied-buffer host-pinning attempt saved its profile but was interrupted with SIGINT/exit 130 under severe host pressure, in an Accelerate connector transfer. Its log is `logs/ltx-baseline-torch-pinning-interrupted.2026-10-16T205747Z.log`; no passing benchmark is attributed to it.
Public `host_register` avoids copied pinned allocations; both completed FT arms use it with unchanged blocks/weights/checks, and actual allocation errors still propagate.

Long-lived containers twice lost CUDA/NVML access while host access remained healthy. A pre-load failed attempt is retained in `logs/ltx-baseline-container-device-failure.2026-10-16T2105Z.log`.
Fresh same-image per-arm containers used NVIDIA runtime plus explicit existing GPU device nodes, unlimited memlock/64 MiB stack and the image's login-shell initialization.
This follows [NVIDIA's per-container mitigation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html#containers-losing-access-to-gpus-with-error-failed-to-initialize-nvml-unknown-error), not a host daemon/driver modification; the external trigger was not established. Low-disk-space warnings remained enabled.

## Historical MR interpretation

Read-only verified heads: !343 `d32ca849`, !359 `4d440c56`, !448 `11395c1e`, !449 `fff0a135`. These workloads are not interchangeable evidence.

| MR | Established evidence | Important limitation |
|---|---|---|
| [!343](https://gitlab-master.nvidia.com/dl/JoC/bermuda/ai-tune/-/merge_requests/343) | 49 frames, 1280x704; five warmups/50 runs; 20.488 to 17.768 s; 45.790 GiB allocated; 48 blocks/192 routes; replan | Eager-versus-compiled output-value validation absent |
| [!359](https://gitlab-master.nvidia.com/dl/JoC/bermuda/ai-tune/-/merge_requests/359) | Fresh-worker benchmark workflow | Explicitly no eager output-value comparison |
| [!448](https://gitlab-master.nvidia.com/dl/JoC/bermuda/ai-tune/-/merge_requests/448) | Pure FT completed 45 s; separate 49-frame resident `proj_in` smoke selected eager, bit-exact but 15.3% slower | No 45-second all-block combined result |
| [!449](https://gitlab-master.nvidia.com/dl/JoC/bermuda/ai-tune/-/merge_requests/449) | 1081 frames / 45.042 s, 1920x1088; all 48 blocks/96 routes; retained strategy; 6141.440 to 4530.608 s and 95.579 to 65.309 GiB sampled device peak | Zero warmups/one run; strict short/full video/audio gates failed; perceptual acceptance not established |

Local evidence is under `/tmp/aitune-gitlab-449.INpwrU/`: !343 `RESULTS.md:10,35,57`; !359 README:151; !448 README:338-365; !449 README:379-445.
!449 reports video maximum difference 77 uint8 levels and audio 0.552856; its README calls 1.355544x a raw ratio, not correctness-accepted acceleration. The numbers do not identify their cause or decide human visual acceptance.

!449 provider `b94be2f813f18f347f222583ca3b536c2bfef20a` [already contained the busy-slot guard](https://github.com/ai-dynamo/flextensor/blob/b94be2f813f18f347f222583ca3b536c2bfef20a/src/flextensor/core/transfer.py#L88). There is no evidence that FlexTensor 0.5.0 newly introduced that guard.
The [!449 adapter, lines 692-721](https://gitlab-master.nvidia.com/dl/JoC/bermuda/ai-tune/-/blob/fff0a1352e5daa9be93fa9da8df1909530598695/aitune/torch/jit/flextensor.py#L692) joined transfers before/after tuning and after individual backend calls; the current base removed those joins while still tuning prepared units.
Restoring them is not the supported fix: [tagged provider `loaders.py:1622-1631`](https://github.com/ai-dynamo/flextensor/blob/v0.5.0/src/flextensor/loaders.py#L1622) documents `join_after_forward()` for forward/piece boundaries, not mid-forward replay. The shell/compute split avoids re-entering that boundary.
Separately, [!449 adapter:46-77](https://gitlab-master.nvidia.com/dl/JoC/bermuda/ai-tune/-/blob/fff0a1352e5daa9be93fa9da8df1909530598695/aitune/torch/jit/flextensor.py#L46)/README:197 supported retained view profiling, while base adapter:33-34 rejected `compile_fn`; the new adapter restores that public workflow without confusing it with replanning.

The published !449 command used a shorter setup video. `benchmark.py:1453-1457` calls `configure_aitune(inductor_dynamic=True)`; default `correctness_only=False` uses actual eager/candidate profiling and enabled MinLatencyStrategy evaluation (`outpaint.py:359-367`).
An unused profiling-disabled helper branch does not prove the published run disabled profiling. Its long workload was separately executed and numerically compared; no offload-causality claim follows from its failed numerical gate.

## Completed larger short workload

Separate artifacts are under `tuning/ltx23_outpaint_compute/`: 17 frames at 24 FPS, 480x288 source, 768x480 stage 1 and 1536x960 final. The same pinned weights, conditioning, schedules, all 48 whole blocks and ordinary AITune checks/profiling remain in use. Each arm is a fresh process with three warmups and ten measured generations.

| Matched measurements | Resident eager | FT-only, profile reuse | AITune + FT, profile reuse |
|---|---:|---:|---:|
| Mean generation seconds | 25.5787736801 | 26.6997030467 | 23.9448304623 |
| Peak allocated GPU bytes | 47,655,917,056 | 34,174,113,280 | 34,170,387,968 |
| Peak reserved GPU bytes | 48,706,355,200 | 35,251,027,968 | 35,221,667,840 |
| Sampled whole-device peak bytes | 49,679,106,048 | 36,272,013,312 | 36,244,750,336 |
| Sampled process RSS peak bytes | 11,948,306,432 | 25,820,540,928 | 26,743,115,776 |
| Setup seconds | 51.300 | 78.163 | 336.334 |

Combined is 1.11505x FT-only throughput (10.3% lower mean latency) and 1.06824x eager throughput (6.4% lower latency), with 28.3% less peak allocated GPU memory than resident eager. The fresh FT-only range is 26.4184-27.0371 s; combined is 23.0289-25.1600 s and trends downward. These are sequential single-machine batches, not confidence bounds or a guarantee across workloads/hardware. This geometry shows a more useful observed benefit than the default tiny case.

The first profile-building FT-only run also passed (715.703 s setup, 27.229509 s mean), but drifted from 28.7484 to 25.7263 s. Before the combined result was known, we chose the fresh `--reuse-profile` baseline control to give both FT arms the same loading path. The original builder result/media remain in `baseline-profile-builder.2026-10-16T220503Z/`; final `comparison.json` uses the fresh 26.699703 s control, not the more favorable builder mean.

The geometry-specific profile SHA256 is `7e35b5c92b27ef28dd41cb02d4b9f00445f87005e597f76ad38d88bfcf548f7d`, verified unchanged before/after combined and the final control (`logs/profile-{before,after}-combined.sha256`). It was calibrated with host-register pinning from the start, unlike the retained tiny profile. Both FT arms report all 48 units, 803,340,288 rolling-block plus 25,000,368,640 unmapped bytes, total 25,803,708,928 managed bytes. No whole-device fraction cap is inferred.

All 48 unique whole-block targets selected real Inductor, with zero failed builds, positive eager/candidate throughputs and min/max video token shapes 2160/4320 covering both stages. All final runs and `compare` exited 0. The independent audit confirmed matching weights/workload/versions/compiler settings/FT policy and the actual fresh-control comparison.

Held-out FT-only and eager video latents/audio are bit-identical, maximum difference zero. Combined versus either reference has video max absolute difference 0.1919990, relative L2 0.0134473 and cosine 0.9999176; audio max 0.1523438, relative L2 0.3842758 and cosine 0.9267873. Both explicit 1e-2 allclose diagnostics fail; all values are finite. Float32 cosine reduction can slightly exceed 1 for identical large tensors; raw metrics remain unmodified and are not quality gates.

## Review artifacts and reproducible commands

All seven final encoded clips decode correctly: 17 frames, 24 FPS, 0.7083 seconds and expected geometry (`media-validation.json`). They are video-only; generated audio is retained in held-out tensor artifacts for numerical diagnostics. Source/eager/combined sampled frames were inspected; eager and combined look very similar, but native generation changes source details. Human quality/source-preservation acceptance remains open.

- [Source clip](/home/kkleczewski/repos/aitune/tuning/ltx23_outpaint_compute/source.mp4)
- [AITune + FlexTensor outpaint](/home/kkleczewski/repos/aitune/tuning/ltx23_outpaint_compute/combined/outpaint.mp4)
- [Resident eager reference](/home/kkleczewski/repos/aitune/tuning/ltx23_outpaint_compute/eager/outpaint.mp4)
- [FT-only reference](/home/kkleczewski/repos/aitune/tuning/ltx23_outpaint_compute/baseline/outpaint.mp4)
- [Independent-seed combined clip](/home/kkleczewski/repos/aitune/tuning/ltx23_outpaint_compute/combined/heldout_outpaint.mp4)
- [Final comparison JSON](/home/kkleczewski/repos/aitune/tuning/ltx23_outpaint_compute/comparison.json)
- [Container, validation and benchmark commands](/home/kkleczewski/repos/aitune/work_logs/flextensor/validation_commands.md)

The command manifest includes the tested local image, login/PYTHONPATH/device setup, native variants, CPU scope, benchmark/control commands and static checks. Reconstructed equivalents and the unrecovered original dependency-install argv are explicitly labeled; a fresh-base bootstrap was not invented or claimed tested. The existing local image and task caches remain available.

## Local commit handoff

All required implementation, native tests, real benchmark/control comparisons, artifact checks and final static checks are complete. This change is delivered as one local Conventional/DCO commit, intentionally unsigned; use `git log -1 --format='%H%n%G?%n%B'` for its identity, `N` signature status and sign-off. The explicit unsigned request overrides the commit skill's usual signing rule. Signing configuration and published history remain untouched. The command is `git commit --no-gpg-sign --signoff -m "fix(jit): isolate flextensor residency from compute tuning"`. Nothing was pushed.

After reviewing the unsigned local commit, the user's manual commands are:

```bash
git commit --amend --no-edit -s -S
git log -1 --show-signature
git push origin HEAD:kkleczewski/jit-flextensor
```

Signing/pushing is left to the user; no force push is requested. Prior chronological detail is preserved in timestamped `work_logs/flextensor/` copies, including byte-verified pre-reconciliation snapshots `2026-10-16T213927Z`.
