<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlexTensor integration improvement plan

Updated: 2026-10-16 21:33 UTC. Status: native matrix and complete tiny LTX comparison passed; larger short-workload experiment being prepared.

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
5. [x] Strengthen real GPU functional coverage: eager/deferred, both known shapes and held-out values, normal correctness, real profiling, no partial fallback, actual offloading, and preserved live compiled dispatch after replan. Final five-case matrix passed with persistent public telemetry.
6. [x] Check wrapper signatures, hooks, structured outputs, selection order, alias/nesting rejection, invalid provider state, and retained-profile mode in focused contract tests (real provider matrix still in progress).
7. [ ] Finish static checks and final artifact reruns; JIT020's four variants and relevant standalone CPU suites already passed, with a source-diff audit proving no core behavior/config changes.
8. [x] Establish a feasible real pretrained LTX 2.3 mini workload on the A6000: short clip and low resolution, all 48 transformer blocks, fixed weights/settings, caller-owned compiler limits, and reproducible dependencies. Resident eager inference completed.
9. [x] Complete matched tiny eager/offload/combined measurements and numerical diagnostics: all 48 real Inductor blocks, 29.3% allocated-memory saving versus resident eager, but only a marginal 2.4% speed gain versus FT-only.
10. [ ] Inspect actual combined video and supply playable source/outpaint links, with numerical and perceptual limitations.
11. [ ] Run one higher-compute short experiment in a separate work directory (768x480 stage 1, 1536x960 final, 17 frames) to test whether the tiny case hides useful compute gains. Preserve the completed tiny results; use the same weights/checks and a new geometry-specific profile.
12. [ ] Finish provenance/reproduction evidence, final static checks, readable report, unsigned DCO commit and manual signing/push instructions. Review all requirements before completion.

## Validation rules

- Functional tests must detect actual feature failures, not mask them with test-side provider-state changes, fake compilation, reset, omitted targets, permanent resident weight copies, or weakened tolerances.
- Ordinary AITune correctness checks remain enabled. Strict toy-model comparisons remain strict. LTX numerical differences are reported as diagnostics alongside finite/structure checks and final manual visual review; do not claim numerical equivalence that was not demonstrated.
- Real profiling must be observed, not inferred from configuration. A performance gate may legitimately select eager; distinguish that from compilation failure/native fallback.
- Budget accounting distinguishes managed weights/rolling blocks from activations, compiler allocations, allocator reserve, and whole-device occupancy. Do not claim a cap from an assertion that only proves some offload.
- Each functional variant runs in a fresh interpreter. Normal workflow registers once and uses complete model traversals, without reset or rollback.
- Do not push or alter PR metadata. Revisit local commit/signing requirements before committing.

## Current constraints and risks

- GPU: RTX A6000, 49,140 MiB. Host RAM: 31 GiB, swap: 14 GiB. Free disk initially about 95 GiB. Task-owned pretrained inference weights and encoded prompt are now cached. Lower resolution reduces activations, not the 22B model weight footprint; keep setup RSS and GPU occupancy visible.
- Wrapper insertion adds a `compute` component to module/state-dict paths. This is an explicit integration topology choice; preserve original module objects/hooks and document serialization limitations instead of inventing key-remapping machinery.
- Prior MR !449 used the same busy-slot guard but an unsupported mid-forward join workaround and a retained-profile path. Its reported speed/memory improvement was not strict numerical acceptance. Neither that workaround nor a lazy-only feature reduction should be ported as the fix.
- The strengthened native test initially exposed the known provider budget discrepancy before tuning. Preserve the strict half-weight total residency cap (369,188,864 bytes), but explicitly reserve the excluded resident head (67,125,248 bytes) from the provider strategy fraction. The actual new native replan result is 335,724,544 managed bytes, below that unchanged cap. Record both strategy request and actual total rather than claiming that a provider fraction alone enforces a physical cap.
- LTX prompt encoding succeeded separately on CUDA. Its newly downloaded text-weight blobs were removed only after validating the reusable prompt artifact, freeing about 45.4 GiB. Real inference weights are downloaded; resident eager generation passed. Provider offload/pinning feasibility is still being measured.
- The first development container lost CUDA/NVML device access while the host and a fresh probe remained healthy. Continue in replacement container `aitune-ft-dev-20261016b` with the same dependencies and workspace, using the image's forward-compatibility shell initialization. Do not attribute this environment incident to model or integration behavior.

## Decisions from the first iteration

- Adapter CPU contract tests: 90 passed. Standalone core regression suite: 668 passed, 79 explicit CUDA-required skips. Core source/API/configuration hashes remain unchanged.
- Real-provider eager/deferred diagnostic replan and deferred retained-profile cases passed all ten Inductor targets. The enabled performance gate also passed with measured eager selections; genuine unsupported-fullgraph fallback passed separately. Final reruns will persist public tuning JSON per variant in the healthy replacement container.
- The functional test's first telemetry check used Python object IDs incorrectly. Corrected it to join public registration report IDs to public tuning-data IDs; no production behavior changed to satisfy that assertion.
- Actual public-domain source decoding/encoding passed: 17 frames at 24 FPS, with a checked exact-zero reference canvas. Resident eager generation measured 13.052329 seconds mean over three runs. Rerun eager after the newly added shared output/provenance checks, then compare like-for-like offloaded and combined modes.
- Do not require provider `unmapped_tensors_bytes` to equal the excluded head: it includes every reachable CUDA tensor absent from the view mapping. Assert the public component sum and unchanged strict total cap instead.
- Real LTX provider startup is expensive on this host: default identity-compile view profiling requires one discovery plus three unmeasured compile warmups plus ten measured root forwards. File-backed weights exceed RAM, causing hundreds of GB of checkpoint reads without requiring a second whole-model CPU clone. The profile is now saved and plans about 13.65 GiB of offload; steady-state timing remains pending.
- Record the baseline profile digest before combined loading, reuse it explicitly, and verify it afterward. Comparisons now reject mismatched workload/versions/compiler/measurement policies, mismatched FT memory/finalization/profiling settings, missing all-48 boundaries and absent actual offload.
- Repository pre-commit checks passed on the changed files. Both independent adapter and documentation reviews reported no actionable findings. Metadata validation passed: 92 scripts, 14 projects, zero errors.
- The default copied-buffer host-pinning experiment was interrupted deliberately after saving its complete profile: host RAM approached saturation, swap was active and inference spent minutes transferring the connector instead of providing useful GPU measurements. Preserve its log and profile; do not count it as a passed benchmark. Retry both FT arms with public `pinned_memory_mode="host_register"`, checking public runtime capability before loading weights. This changes only caller allocation policy, not provider/core code, weights, block coverage or selected profile strategy.
- Use the same `TORCHINDUCTOR_COMPILE_THREADS=1` environment in all LTX arms and record it. Increase common measurements to three warmups and ten timed generations before judging gains. Reuse the existing complete profile for both FT arms and check its digest afterward; its profiling originated in the copied-buffer attempt, so do not claim a new host-register profile calibration.
- Matched references completed: resident eager 11.614826 seconds mean and 47,331,982,848 peak allocated bytes; FT-only 21.725765 seconds and 33,443,462,656 peak allocated bytes. FT held-out video latents/audio are bit-identical to eager. Both strict provenance and actual-offload checks passed, and the saved profile digest stayed unchanged.
- A second long-lived-container access loss occurred before the next model load. Preserve that failed log; continue in fresh same-image containers with explicit existing GPU device nodes and the NVIDIA runtime. This is a per-container documented mitigation, not a host daemon/driver change. The exact external trigger was not established.
- Real combined setup has selected all 48 Inductor graph routes with successful candidate builds, correctness and actual eager/candidate throughput telemetry. Setup completed in 342.874 seconds; serving warmups/measurements/held-out outputs and final comparison remain unverified.
- Preserve ephemeral JIT build logs before disposable-container exit. Future example runs now set the public JIT cache directory explicitly below the task work directory; the first combined run used the image's default JIT cache, copied to `combined/jit-build-cache/`.
- The tiny comparison completed: eager 11.614826 s, FT-only 21.725765 s, combined 21.208194 s (three warmups/ten runs). The 2.4% mean improvement is smaller than observed run drift, so it is not a convincing speedup result. Both FT arms peak at 33,443,462,656 allocated bytes versus 47,331,982,848 resident eager.
- Tiny FT-only held-out video/audio are bit-identical to eager. Compiled outputs are finite but not numerically equivalent; compare JSON records the differences. Actual source, reference and combined videos exist. Manual quality acceptance is still the user's decision.
- Add only caller width/height controls to the example, preserving its original defaults. A new 768x480 experiment will reuse model files and the resolution-independent encoded prompt, but regenerate source and FT profile. Do not change conditioning, drop blocks or relax validation to obtain a speedup.

## History

Timestamped copies of this file and `report.md` live in `work_logs/flextensor/`. Archive before material plan/status revisions, and record why the next action changed.
