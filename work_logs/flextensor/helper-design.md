<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Additive FlexTensor preparation helper

## Requirements and design

Implement `prepare_flextensor_units(model_or_owner, *, ft_config, target=None,
name="default", load_profile_from=None)` in the integration layer. Return an immutable
`PreparedFlexTensor` containing the exact FT-returned `model` proxy, `manager`,
`effective_config`, and ordered `unit_paths`. A supplied dotted `target` is resolved
from its owner using ordinary public attribute access and assigned using ordinary
`setattr`; without a target, the caller uses the returned proxy. Do not infer parents.

The current `FlexTensorUnit`, manual FT setup, registration, and replan APIs remain
available and unchanged. Standalone AITune and FT source/dependencies are not modified.
New code calls only public Torch, FT, and AITune APIs. FT is imported lazily on use.

Preparation is setup only: no forwards, startup loops, device moves, registration,
tuning, profiling, compiler-limit changes, warmups, replan, recovery, or reset.
Retain existing qualified mode choices: external compilation uses `compile_fn=None`;
retained view profiling uses the identity callback. Fail unsupported modes before
model mutation. Load profiles only through explicit public `offload_from_profile`.

### Selection and residency preservation

Resolve the configuration against the original, unwrapped public Torch module and
parameter tree. A small integration-owned matcher implements FT 0.5.0 documented
name/class/parameter selection semantics; it does not import FT selector utilities.
Module selection uses single-segment standalone stars, parameter/exclude matching
uses recursive standalone stars, and class matching uses short and qualified names.
Only `*` and `?` are special, not shell bracket expressions.

Produce two independent results: ordered outermost include candidates followed by
unit exclusion, and the global per-parameter residency mask including root parameters.
The ancestor guard runs before unit exclusions. Insert shells only at final selected
unit paths. Retain original include patterns and append exact original outermost
candidate paths, including those later excluded, to preserve their ancestor guards. Replace
exclusions with exact dropped candidate paths and every original kept parameter path,
mapping `.compute` only under selected shells. Use explicit `name:` prefixes.

Preflight the generated configuration against a virtual wrapped tree before mutation:
require the same ordered unit boundaries and identical global residency mask. This
avoids broadening parameter policies, losing class selectors, changing excluded
ancestors, or introducing boundaries from root/outside-unit parameter selectors.
Keep the caller's configuration untouched using validated shallow `model_copy`;
preserve unrelated fields and custom strategy identity.

Reject empty selection, existing shells, boundaries without an executable forward,
unrepresentable literal wildcard/whitespace names, and ambiguous cross-boundary aliases
before mutation. Use the existing manual path for unusual unsupported model structures;
do not silently change their policy. Ordinary wrappers are not attribute/type-transparent.
Validate target/name/config/mode and public active-manager state before mutation.
Public state cannot prove a name was never initialized: require a fresh named manager.
Provider/setup/setter failures propagate; no rollback is promised after FT starts.

## Design review

Independent read-only selection and lifecycle reviews support the design. Selection
review corrected an unsafe earlier encoding: adding exact globally-offloaded parameter
includes could create new unit boundaries. Retaining original includes avoids that
change; the final review also requires exact excluded candidates in the includes to
preserve ancestor suppression after prefix derivation changes. Virtual-policy verification
remains mandatory. Lifecycle review requires the
actual FT proxy, generic owners, normal assignment semantics, lazy FT import, explicit
fresh-manager lifetime, and no hidden execution. These findings were incorporated
before production implementation started.

## Implementation plan (for independent review before coding)

1. Add a small internal integration-only selector/planner using public Torch metadata.
2. Add the public result and helper beside the existing public FT adapter APIs. Keep
   existing registration/replan implementation and AITune core files unchanged.
3. Add dedicated helper unit tests with a strict public-surface provider fake: target
   installation, identity/config preservation, selector policies, errors, profile calls,
   lazy optional import, and no hidden forwards or tuning.
4. Preserve existing manual functional variants and add helper variants to the same
   native acceptance assertions. Add public FT black-box selector/profile-reload tests.
   Run JIT020's four variants and native-failure JIT022 as regressions.
5. Migrate LTX setup to the owner-aware helper, preserving app-specific placement and
   all lifecycle operations. Validate examples through functional workflows, not unit
   tests of example code.
6. Run focused plus broad relevant unit suites, GPU functional matrix, Ruff/format,
   pre-commit, functional metadata validation, pytype, and source-invariance checks.
7. Apply approved documentation edits, update PR #89's description accurately, and
   independently review feature, UX, and all three stack layers (#43 -> #45 -> #89).
   Do not sign, push, rewrite commits, or change signing settings.

## Documentation Phase 1 report

The docs skill's referenced docs-auditor instructions are absent; an independent
read-only fallback scanned all 49 Markdown guides/recipes/root README and navigation,
and reviewed FT/JIT content in depth. No broken local Markdown links were found.
This is not execution of unrelated documentation snippets.

Proposed in-scope edits (user approval required by the documentation skill):

- `docs/guides/flextensor.md`: helper-first root/pipeline setup, return fields, original
  selector namespace, copied config, unchanged residency, public-only API, no hidden
  lifecycle, explicit loading/failure rules, updated verification; retain manual setup.
- `examples/LTX23OutpaintAITuneFlexTensor/README.md`: short `target="transformer"`
  setup and unchanged placement/startup/registration/tuning/warm/replan sequence.
- `docs/guides/jit_tuning.md` and `docs/learn/quick_start.md`: mention optional FT
  preparation in existing FT links; no standalone behavior/documentation changes.

An optional preexisting LTX navigation entry is outside this feature and is not planned.
No unrelated broad documentation rewrites are proposed.

The author approved all four edits on 2026-10-18. They are now applied: helper-first
root/owner setup, the preserved manual alternative, explicit profile/lifecycle contracts,
and updated JIT/quick-start links. Independent final documentation/API/UX review approved
all four files; its weight-copy wording clarification was incorporated. All 42 Python
and seven Bash snippets in these files pass syntax checks; all 18 local links and seven
heading references resolve. Syntax checking is not execution of those snippets.

## Verification status

Design and implementation plan independently reviewed and approved before production
coding. Plan review added explicit excluded-ancestor policy cases, fresh-interpreter
profile roundtrips, native LTX two-stage tuning/held-out execution, and separate reviews
of current PR heads/diffs/descriptions. Implementation and validation are complete;
changes remain uncommitted on HEAD `0a0ff88514f84619063398613bcfab53333ff94b`.
GPU is available with elevated
read-only access (RTX A6000, driver 570.211.01); sandbox device/socket denials are not
hardware failures. Existing validation image and caches remain available.

Independent source review found no new helper correctness issue and judged the API
coherent. It confirmed that existing manual registration cannot prove preparation
provenance: a shell installed after FT preparation can pass structural checks while its
compute still contains FT scheduling. The helper enforces the correct order for its fresh
model contract; the guide's universal-rejection claim must be narrowed, not replaced by
private provider inspection or a heuristic guard. Existing adapter dispatch checks use
AITune internals; new helper code adds no such dependency.

Independent lower-stack review found no actionable defect in #43/#45. Its explicit
lifetime/API changes (fixed mode and no reset) are intentional preexisting changes, so
only the new helper's standalone invariance is claimed. All accepted external strategies
have preflight unit coverage; the native functional matrix is primarily OneBackendStrategy
with Inductor, not an all-strategy GPU qualification.

## Implementation and UX

Implemented `PreparedFlexTensor` and the public setup helper in the optional adapter,
with the integration-owned planner in `_flextensor_preparation.py`. LTX now uses:

```python
prepared = prepare_flextensor_units(
    ltx_model,
    target="transformer",
    ft_config=cfg,
    name="ltx",
    load_profile_from=profile_dir if reuse_profile else None,
)
manager = prepared.manager
```

No subsequent `ltx_model.transformer = prepared.model` assignment is needed. Bare-root
callers still use `model = prepared.model`, since a function cannot rebind a caller's
local variable. Dotted targets work for ordinary owners and module owners through
normal setters; unrelated aliases are not rewritten. The helper preserves the original
container when replacing its selected children.

UX review: one call replaces the wrapping loop, callback selection, FT setup, manager
lookup and optional parent assignment. Explicit config and target make policy and
mutation visible. Returned original unit paths and effective config make the result
inspectable without retaining stale unit references. Existing startup, one registration
batch across managers, representative recording, deferred activation, warmup and optional
replan remain visible: preparation does not pretend to be a complete tuning pipeline.
The original manual API is retained for advanced setup; there is no new coordinator,
reset, retry state machine, backend restriction, or core configuration change.

The LTX partial-NVML-initialization review finding was also corrected: RSS sampling
continues if handle lookup fails, and a successful init is shut down on that partial
failure. No unit tests were added for example code.

## Validation evidence (2026-10-18)

Local logs are under `tuning/ft-helper-20261018/logs/` (ignored runtime artifacts, not
files expected to exist on GitHub). Tests use the prepared Python 3.12 / Torch 2.10
environment. GPU workers use local image `aitune-flextensor-dev:20261016b`, ID
`sha256:7c4e74111b4383d0674467de8979071f5f1e0dd5875d78762a0b598013db8145`, RTX A6000,
driver 570.211.01 and the image's CUDA forward-compatibility initialization.

- Dedicated helper unit suite: **82 passed**, six existing TorchAO deprecation warnings.
- Existing adapter suite: **90 passed**, six existing TorchAO deprecation warnings.
  The initial shell observation timed out while the Docker child continued; its live
  process was verified and the complete passing summary was subsequently retained.
- Same 43-file core regression scope plus test-isolation utilities: **692 passed,
  79 explicitly CUDA-required skips**, 64 warnings; command exit 0. This extends the
  earlier 668-pass scope by 24 utility tests, without dropping baseline cases.
- All eight JIT021 cases passed: original manual four plus helper four, using identical
  assertions. Diagnostic cases select all ten real Inductor targets, run ten ordinary
  correctness checks and 600 profiling measurements, preserve known/held-out shapes,
  and report 335,724,544 managed bytes under the 369,188,864-byte independent cap.
  Replan modes finish done/applied; retained modes do not replan. Enabled performance
  validation legitimately selects measured eager winners in both setup paths.
- JIT020's four original prepared-unit/placement cases passed in fresh GPU workers.
- Final manual/helper native compiler-failure JIT022 variants both passed with every
  prior correctness, hooks, residency and no-replan assertion retained. The complete
  unique native matrix is 14 variants: JIT020 four, JIT021 eight, JIT022 two.
- Public-only JIT023: **all five policies passed in 17 fresh GPU interpreters**, including
  strict two-shape numerical outputs, real rolling storage, matching public unit names,
  parameter storage-sharing groups and managed-memory components. Helper-created
  profiles reload in fresh workers; a manual-shell profile also reloads through the
  helper. Actual FT proxy assignment on a real `nn.Module` owner is exercised.
  The initial outside-unit baseline failed before reaching the helper because a direct
  caller-owned parameter remained on CPU. The fixture now places it identically for
  both paths; no helper/provider fix, weakened cap or full-residency workaround was used.
  Both the failed initial log and passing `public-policy-profiles-v2.log` are preserved.
- Ruff, format, configured task-file pre-commit, functional metadata (93 scripts /
  14 projects / zero errors), focused pytype (two sources / 127 local dependencies),
  whitespace and standalone source-invariance checks passed.
- Installed FT source tree compares byte-for-byte with the clean v0.5.0 checkout at
  `2561216341d1258d323fa84e5b0cd30e29765dd4`, excluding Python caches and built `.so`
  artifacts. No provider files, dependencies or lower-stack production files changed.

The migrated all-48-block LTX two-stage helper/profile-reuse run passed. All 48 targets
selected real Inductor, with state counts `{"tuned": 48}`. Stage 1 was 256x160, stage 2
512x320, and both measured/held-out videos contain 17 RGB frames. Independent-seed
video latents and audio are **bit-identical** to the earlier manual-shell combined run
(maximum absolute and relative differences zero). Managed memory is unchanged at
25,396,992,512 bytes: 762,576,896 rolling blocks + 24,634,415,616 unmapped bytes.

The run uses a separate work directory and copied original profile, SHA256
`784428c8f940be44b3663b2eaf763a2e74b05ce6be15273444309e2e1c61c787`; old benchmark
artifacts remain intact and the original/copied profile digests remained unchanged.
Logs are `ltx-helper.log` and `ltx-heldout-helper-vs-manual.log`; output metadata and
media are local under `tuning/ft-helper-20261018/ltx/combined/`. One warmup / one
measured generation is functional smoke
validation, not new comparative performance evidence or a perceptual-quality gate.

## Completion audit

1. Design: recorded above and independently reviewed before implementation.
2. Implementation plan: recorded above and independently reviewed before coding.
3. Public APIs: the new helper/planner use public Torch and FT APIs and the public
   integration shell. No new private core/provider call; preexisting adapter dispatch
   checks remain unchanged. Standalone production-source invariance was checked.
4. KISS: one setup call plus a policy planner; no hidden lifecycle coordinator, reset,
   parent inference, retry mechanism or new core configuration.
5. Manual option: existing APIs and original manual functional variants remain;
   the guide retains an explicit manual alternative alongside helper setup.
6. Documentation and PR metadata: four author-approved public files updated; all three
   descriptions distinguish local helper changes from their published heads.
7. Feature/stack review: independent helper, native test and #43/#45 reviews completed;
   no remaining actionable defect. The manual late-wrapping limitation is documented.
8. UX: parent-aware and bare-root paths reviewed, exact proxy installed, selectors and
   mutation explicit, startup/registration/tuning/finalization kept visible.
9. Validation: existing relevant unit/functional coverage retained, dedicated helper
   and public FT tests added, 864 CPU passes / 79 explicit skips, 14 native variants,
   17 public-policy workers, and all-48-target two-stage LTX smoke/held-out parity.

The docs-only finish does not rerun expensive GPU workflows or turn the prior functional
smoke into a performance claim. Final configured pre-commit/static checks cover the
complete local feature and approved docs. All configured task-file pre-commit hooks
passed (`precommit-docs-final.log`); metadata revalidation found 93 scripts, 14 projects
and zero errors (`functional-metadata-docs-final.log`), with the same preexisting
multiple-block warning in `001_aitune_torch_version_test.py`. The short-lived Docker CLI observation timed out;
the container job completed and both final logs were read back, without restarting
the checks. Whitespace and standalone production-source invariance checks also passed.

PR descriptions #43, #45 and #89 have been updated. They explicitly separate published
heads from the uncommitted helper, preserve historical benchmark limitations, and record
the current validation without claiming unpublished CI results. Lower descriptions
confirm this helper adds no commit/dependency/core change to their branches.

No new commit, signing, push, history rewrite or signing-configuration change has been
performed. PR #43/#45 heads remain unchanged; all new implementation belongs to #89.
