<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlexTensor integration improvement plan

Updated: 2026-10-16 22:17:50 UTC. Status: adapter/native, tiny and larger combined passed; final fresh FT-only profile-reuse control active. No commit or push yet.

## Objective and fixed scope

Improve PR #89 on top of #43 and #45 using unmodified FlexTensor 0.5.0. Standalone AITune keeps its API, defaults, configuration/cache identities, per-module compilation, warmup, correctness, profiling, performance evaluation and fallback behavior. The initial integration backend is TorchInductor JIT; do not add lazy-only core behavior, reset, rollback, repeated registration or provider-private coordination.

Use all 48 real pretrained LTX 2.3 whole transformer blocks, not a quantized/random/reduced proxy. Measure rather than assume a performance improvement. Supply actual source/outpaint videos for human review; numerical diagnostics do not substitute for perceptual acceptance.

## Implemented architecture

Insert an integration-owned outer `FlexTensorUnit(original_compute)` before provider preparation. FlexTensor controls that shell's residency boundary; unchanged explicit external-placement registration targets only its original `compute` child after discovery/profiling reaches inference. Ordinary repeated AITune compute calls stay inside one provider call.

Resolve final live shells through public APIs. The registration retains immutable manager/model/shell/compute/forward associations for explicit whole-model replanning. Support the separate identity-`compile_fn` retained-view-profile path without calling or claiming a replan. Wrapping adds `compute.` to parameter/checkpoint paths; do not invent checkpoint remapping or broad attribute delegation.

## Completion checklist

1. [x] Verify stack heads, preserve unrelated work and the earlier unpushed lazy-only experiment.
2. [x] Inspect guidance/skills, historical MRs and provider ownership contracts. Preserve the no-new-0.5.0-guard-regression conclusion.
3. [x] Retain fresh supplemental exact-base eager/deferred failures and provider-supported whole-model lazy success. Label them current-runtime evidence, not retroactive proof of pre-dependency-change chronology.
4. [x] Implement only the provider-specific production adapter; verify standalone source/API/configuration/cache invariance.
5. [x] Cover signatures, hooks, outputs, training flags, ordering, ownership/nesting rejection, pending/fallback states and retained-profile contracts: 90 adapter tests passed.
6. [x] Run JIT020's four native variants and final JIT021/JIT022 five-case matrix in fresh interpreters; retain logs/public telemetry. Standalone CPU suite: 668 passed, 79 explicit CUDA-required skips.
7. [x] Establish the real two-stage tiny LTX workload, all 48 whole blocks, ordinary correctness and real eager/candidate profiling, fixed weights/LoRA and caller-owned finite compiler limits.
8. [x] Complete matched tiny eager/FT-only/combined runs and comparison: three warmups/ten measurements per arm; record setup, steady-state memory and numerical diagnostics separately.
9. [ ] Deliver final playable source/combined/held-out video links. Actual tiny frames were inspected; source appearance changes already occur in eager, and human quality acceptance remains open.
10. [ ] Finish the separate higher-compute short experiment: 768x480 stage 1, 1536x960 final, 17 frames. Eager passed at 25.578774 s; profile-building FT-only passed at 27.229509 s and is archived. Combined passed at 23.944830 s with all 48 real Inductor routes, but also drifted downward. The pre-planned fresh baseline `--reuse-profile` control is active; finish its comparison before quoting gains. Profile digest stayed unchanged; builder FT/eager held-out outputs are bit-identical, compiled values finite but numerically different.
11. [ ] Reconcile final reports/artifacts, archive before material changes, then rerun final static checks after report edits. Existing pre-commit passed; functional metadata: 92 scripts, 14 projects, zero errors.
12. [ ] Prepare the requested unsigned Conventional/DCO commit, verify no signature, and provide exact later manual signing/push commands. Do not sign, push, change signing configuration, rewrite published history or edit PR metadata.

## Verified tiny result and interpretation

| Three warmups / ten runs | Resident eager | FT-only | AITune + FT |
|---|---:|---:|---:|
| Mean generation seconds | 11.6148258705 | 21.7257649820 | 21.2081939233 |
| Peak allocated GPU bytes | 47,331,982,848 | 33,443,462,656 | 33,443,462,656 |

The combined mean is only a marginal 2.4% latency reduction versus FT-only, within observed run drift; do not present it as strong acceleration. Resident eager is faster. Both FT modes save 29.3% peak allocated GPU memory versus resident eager. Faster individual compiled blocks are not proof of an end-to-end gain.

Tiny FT-only held-out video latents/audio are bit-identical to eager. Combined values are finite but fail the explicit 1e-2 allclose diagnostic; report the differences, not numerical equivalence. Actual eager and combined frames look very similar, but both change source expression/appearance. No perceptual acceptance is claimed.

## Validation and evidence rules

- Preserve strict toy-model tolerances and all-target/actual-offload assertions. Do not fake compilation, omit units, make every weight resident, manipulate provider tickets or suppress genuine exceptions.
- Keep ordinary correctness enabled. Diagnostic performance validation still performs real baseline/candidate profiling; enabled mode may legitimately select a measured eager backend. Distinguish that from native compiler-failure fallback.
- Assert managed-memory components and a genuinely independent cap in the native test; neither a configured fraction nor a steady-state snapshot proves a whole-device/transient cap.
- Native tests retain the unchanged 369,188,864-byte total cap. Explicit resident-head reservation produces a 302,063,616-byte strategy request and 335,724,544 actual managed bytes. Do not equate unmapped bytes with head bytes.
- Every functional variant uses a fresh interpreter and one registration batch. No recovery/reset API or unsupported mid-forward transfer join.
- Compare matched weights, geometry, conditioning, versions, finite compiler settings, warmups/runs, host-pinning and FT policy. Tiny FT arms reuse the same completed profile with SHA256 `784428c8f940be44b3663b2eaf763a2e74b05ce6be15273444309e2e1c61c787`.
- Keep the tiny directory intact. The larger geometry must use its own source/profile/results, despite sharing resolution-independent prompt embeddings and model files.

## Constraints and durable decisions

- Hardware: RTX A6000, 49,140 MiB; host RAM 31 GiB, swap 14 GiB. Frame geometry reduces activations, not the model's weight footprint.
- Default copied host-pinning exhausted useful host headroom after profile creation; that interrupted run is not a benchmark pass. Both completed FT arms use public `host_register` with the same retained strategy; its profile originated in the earlier copied-buffer calibration.
- View startup can reread mmap-backed weights for one discovery, three unmeasured warmups and ten measured root forwards. This is not evidence of a whole-model CPU clone or hang. Public profile absence alone does not locate the phase.
- Device-access failures occurred in long-lived containers, not the host. Preserve failed logs; fresh same-image containers used documented per-container device-node/runtime mitigation without host driver/daemon changes.
- Diffusers ZIP metadata pins `9159a587b1f148a2cc911deb72b6d755508ef38b`; two key installed LTX files were byte-verified against that GitHub commit. Existing result JSONs with null VCS revision remain unchanged.
- Keep low-disk-space warnings and real failures visible. Preserve ephemeral build logs before disposable containers exit; future example runs configure the public JIT cache below the task directory.
- Only task-owned text-weight downloads were removed after successful prompt encoding, freeing about 45.4 GiB; existing user caches were not touched.

## History and next handoff

Current full evidence and historical MR interpretation are in `report.md`. Timestamped prior copies live in `work_logs/flextensor/`; the pre-reconciliation plan/report were byte-verified and archived at `2026-10-16T213927Z`. This revision removes superseded pending claims and records the completed tiny result without assuming larger-workload outcomes.

Next: finish the larger matched experiment, update/qualify its actual result, archive and reconcile again, run the post-report static sweep, then prepare the unsigned DCO commit and final artifact handoff.
