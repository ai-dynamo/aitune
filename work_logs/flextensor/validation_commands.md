<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlexTensor validation command manifest

Commands below are for this checkout on the validation machine. This manifest was assembled read-only from retained logs, result arguments, source CLIs and container inspection; it did not rerun validation. Commands are normalized runnable equivalents unless explicitly identified as retained argv. Actual outcomes and limitations belong in `report.md`; a command here is not proof of a successful run.

## Existing local runtime

The tested local derivative is `aitune-flextensor-dev:20261016b`, image ID:

```text
sha256:7c4e74111b4383d0674467de8979071f5f1e0dd5875d78762a0b598013db8145
```

Its public base was:

```text
ghcr.io/ai-dynamo/aitune/nvcr-torch-26.01-py3@sha256:bd4b0ca692cec0d04dcfab89cb8cd1fc22a593e070c851308361d46a05fa67fb
```

The complete initial dependency-install/build argv was not retained. There is no claim that a fresh base plus the example's dependency ranges recreates this exact image. No speculative bootstrap recipe has been executed. Reuse the existing derivative on this machine; its `/tmp/integration-python` user-site installation is inside the image, not a missing host bind mount.

Observed runtime: Python 3.12.3, Torch `2.10.0a0+a36e1d39eb.nv26.1.42222806`, CUDA 13.1, FlexTensor 0.5.0, Transformers 4.57.6, Accelerate 1.15.0, PyAV 19.0.1 and PEFT 0.18.1. Diffusers 0.39.0.dev0 came from `https://github.com/huggingface/diffusers/archive/9159a587b1f148a2cc911deb72b6d755508ef38b.zip`; `direct_url.json` records archive SHA256 `a4de10c581072337bf25a1641b6876081e43bba4bcc00f1567004f89725e7178`. Two key LTX source files were independently byte-verified; this is not a whole-installation hash. Existing JSON null VCS revisions were not rewritten.

From the host repository root, define this launch helper. Its settings were verified against the active fresh baseline container; only its generated container name and command formatting differ. The NVIDIA entrypoint remains intact; `bash -lc` performs the image's login/forward-compatibility initialization. Explicit existing device nodes were the documented per-container mitigation for lost device access, not a host driver/daemon change.

```bash
cd /home/kkleczewski/repos/aitune
set -o pipefail
ft_worker() {
  docker run --rm --runtime=nvidia --gpus all --ipc=host \
    --device /dev/nvidia0 --device /dev/nvidiactl \
    --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
    --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
    --user 361867322:748400513 --workdir /workspace \
    --mount type=bind,src=/home/kkleczewski/repos/aitune,dst=/workspace \
    --mount type=bind,src=/tmp/aitune-flextensor-integration-split.K2xP8A/container-passwd,dst=/etc/passwd,readonly \
    --env PYTHONUSERBASE=/tmp/integration-python \
    --env PYTHONPATH=/workspace:/tmp/integration-python/lib/python3.12/site-packages \
    --env HF_HOME=/workspace/tuning/ltx23_outpaint/huggingface \
    --env HF_HUB_OFFLINE=1 --env HF_HUB_DISABLE_PROGRESS_BARS=1 --env TORCHINDUCTOR_COMPILE_THREADS=1 \
    --env AITUNE_TESTS_USE_DEVICE=cpu \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    aitune-flextensor-dev:20261016b bash -lc 'exec "$@"' aitune-worker "$@"
}
```

Run one GPU worker at a time. The passwd bind is an existing task-created file; preserve it. Offline Hugging Face mode assumes the already populated task caches. The image's inherited `AITUNE_TESTS_USE_DEVICE=cpu` does not affect the native scripts' explicit CUDA calls, but must be **unset** for CPU-only pytest: the fixture requires CUDA whenever that override is set. CPU commands below unset it and hide CUDA. Do not replace the repository/user-site paths with a different global environment.

## Native functional matrix

Each invocation below starts a fresh container/interpreter. These are source-derived equivalents of the retained passing variants, not recovered historical shell quoting. JIT020's real option is `--device-management`; the underscore alias also works.

```bash
ft_worker python tests/functional/pytorch/jit/020_jit_existing_prepared_units.py --mode=eager --device-management=aitune
ft_worker python tests/functional/pytorch/jit/020_jit_existing_prepared_units.py --mode=deferred --device-management=aitune
ft_worker python tests/functional/pytorch/jit/020_jit_existing_prepared_units.py --mode=eager --device-management=external
ft_worker python tests/functional/pytorch/jit/020_jit_existing_prepared_units.py --mode=deferred --device-management=external

ft_worker env AITUNE_OUTPUT_DIR=/workspace/tuning/ltx23_outpaint/artifacts/eager-diagnostic-replan \
  python tests/functional/pytorch/jit/021_jit_flextensor.py --mode=eager --performance=diagnostic --finalization=replan
ft_worker env AITUNE_OUTPUT_DIR=/workspace/tuning/ltx23_outpaint/artifacts/deferred-diagnostic-replan \
  python tests/functional/pytorch/jit/021_jit_flextensor.py --mode=deferred --performance=diagnostic --finalization=replan
ft_worker env AITUNE_OUTPUT_DIR=/workspace/tuning/ltx23_outpaint/artifacts/deferred-enabled-replan \
  python tests/functional/pytorch/jit/021_jit_flextensor.py --mode=deferred --performance=enabled --finalization=replan
ft_worker env AITUNE_OUTPUT_DIR=/workspace/tuning/ltx23_outpaint/artifacts/deferred-diagnostic-profile \
  python tests/functional/pytorch/jit/021_jit_flextensor.py --mode=deferred --performance=diagnostic --finalization=profile
ft_worker env AITUNE_OUTPUT_DIR=/workspace/tuning/ltx23_outpaint/artifacts/native-fallback \
  python tests/functional/pytorch/jit/022_jit_flextensor_fallback.py
```

Do not execute those artifact destinations over the retained evidence if preserving the original run; choose new per-run directories. Logs already retained under `tuning/ltx23_outpaint/logs/` are `jit020-<mode>-<placement>.log` and `final-<case>.log`; final public JSON is in the corresponding `artifacts/<case>/` directory. Enabled performance selection may legitimately choose measured eager winners; the two required diagnostic/replan cases separately require all ten real Inductor routes. Native compiler-failure fallback is a separate negative case, not an accepted partial-compilation result.

## CPU regression scope

Direct focused equivalent for the 90 adapter tests:

```bash
ft_worker env -u AITUNE_TESTS_USE_DEVICE CUDA_VISIBLE_DEVICES= \
  python -m pytest tests/unit/torch/jit/test_flextensor.py -x -v
```

The exact historical argv for the 668-pass/79-explicit-CUDA-skip standalone run was not retained. Its complete 747-item scope is retained in `tuning/ltx23_outpaint/logs/core-regressions-collect.log`. This reconstructs that equivalent file scope, excluding the separately run adapter test, without guessing broader directories or downloading pretrained models:

```bash
ft_worker env -u AITUNE_TESTS_USE_DEVICE CUDA_VISIBLE_DEVICES= bash -lc '
  mapfile -t core_test_paths < <(
    sed -n "s/^\(tests\/.*\.py\): [0-9][0-9]*$/\1/p" \
      tuning/ltx23_outpaint/logs/core-regressions-collect.log | sort -u
  )
  test "${#core_test_paths[@]}" -eq 43 || exit 1
  python -m pytest -n 4 -x -v "${core_test_paths[@]}"
'
```

The saved scope contains JIT registration/lifecycle, module/signature/recording/dispatch, strategy/correctness/profiling, checkpoint and three selected backend files. `core-regressions-cpu.log` retains actual test names, explicit skip reasons and totals. The four-worker scheduling above is an equivalent runnable reconstruction, not an assertion that every original pytest option was recovered.

## Larger matched workload

Use a separate work directory; retain the completed tiny experiment. Commands below normalize the larger run's recorded arguments. Model weights/cache and prompt embeddings are shared, but source/profile/results are geometry-specific. On the validation machine the 17-frame source was prepared from the actual Helen Hill clip; prompt tensors and their provenance JSON are resolution-independent.

```bash
mkdir -p tuning/ltx23_outpaint_compute/logs
ft_worker python examples/LTX23OutpaintAITuneFlexTensor/outpaint.py prepare-source \
  --work-dir tuning/ltx23_outpaint_compute --model-cache tuning/ltx23_outpaint/models \
  --input-video tuning/ltx23_outpaint/HelenHillInterview.theora.ogv --width 768 --height 480
cp --no-clobber tuning/ltx23_outpaint/prompt.pt tuning/ltx23_outpaint_compute/prompt.pt
cp --no-clobber tuning/ltx23_outpaint/prompt.json tuning/ltx23_outpaint_compute/prompt.json

ft_worker python examples/LTX23OutpaintAITuneFlexTensor/outpaint.py run --mode eager \
  --work-dir tuning/ltx23_outpaint_compute --model-cache tuning/ltx23_outpaint/models \
  --width 768 --height 480 --warmups 3 --runs 10 \
  --gpu-memory-fraction 0.65 --pinned-memory-mode host_register \
  2>&1 | tee tuning/ltx23_outpaint_compute/logs/ltx-eager.log

ft_worker python examples/LTX23OutpaintAITuneFlexTensor/outpaint.py run --mode baseline \
  --work-dir tuning/ltx23_outpaint_compute --model-cache tuning/ltx23_outpaint/models \
  --width 768 --height 480 --warmups 3 --runs 10 \
  --gpu-memory-fraction 0.65 --pinned-memory-mode host_register \
  2>&1 | tee tuning/ltx23_outpaint_compute/logs/ltx-baseline.log

sha256sum tuning/ltx23_outpaint_compute/flextensor-profile/profile.json
```

The profile-building baseline's completed directory was preserved as `tuning/ltx23_outpaint_compute/baseline-profile-builder.2026-10-16T220503Z/` before the control reused the `baseline/` result path. Preserve both results/media when repeating this workflow; do not overwrite that archive. The recorded profile SHA256 before/through combined is `7e35b5c92b27ef28dd41cb02d4b9f00445f87005e597f76ad38d88bfcf548f7d` (`logs/profile-before-combined.sha256`).

```bash
ft_worker python examples/LTX23OutpaintAITuneFlexTensor/outpaint.py run --mode combined --reuse-profile \
  --work-dir tuning/ltx23_outpaint_compute --model-cache tuning/ltx23_outpaint/models \
  --width 768 --height 480 --warmups 3 --runs 10 \
  --gpu-memory-fraction 0.65 --pinned-memory-mode host_register \
  2>&1 | tee tuning/ltx23_outpaint_compute/logs/ltx-combined.log

ft_worker python examples/LTX23OutpaintAITuneFlexTensor/outpaint.py run --mode baseline --reuse-profile \
  --work-dir tuning/ltx23_outpaint_compute --model-cache tuning/ltx23_outpaint/models \
  --width 768 --height 480 --warmups 3 --runs 10 \
  --gpu-memory-fraction 0.65 --pinned-memory-mode host_register \
  2>&1 | tee tuning/ltx23_outpaint_compute/logs/ltx-baseline-reuse.log

sha256sum tuning/ltx23_outpaint_compute/flextensor-profile/profile.json
ft_worker env CUDA_VISIBLE_DEVICES= python examples/LTX23OutpaintAITuneFlexTensor/outpaint.py compare \
  --work-dir tuning/ltx23_outpaint_compute --model-cache tuning/ltx23_outpaint/models \
  --width 768 --height 480 \
  2>&1 | tee tuning/ltx23_outpaint_compute/logs/ltx-compare.log
```

The active control's retained Docker `Config.Cmd` was `bash -lc 'set -o pipefail; python examples/LTX23OutpaintAITuneFlexTensor/outpaint.py run --mode baseline --reuse-profile --work-dir tuning/ltx23_outpaint_compute --model-cache tuning/ltx23_outpaint/models --width 768 --height 480 --warmups 3 --runs 10 --gpu-memory-fraction 0.65 --pinned-memory-mode host_register 2>&1 | tee tuning/ltx23_outpaint_compute/logs/ltx-baseline-reuse.log'`. The helper moves `tee` to the host without changing Python arguments.

Defaults shared by those runs are retained-profile finalization, diagnostic performance validation, three AITune profile steps, ten FT profile steps and finite caller-owned compiler limits 128/512. Results record requested/effective values. Do not compare builder setup time directly with profile-loading setup or call sequential drifting batches a conclusive speedup. Archived float32 cosine diagnostics can slightly exceed 1 from reduction roundoff even for bit-identical tensors; maximum difference zero establishes equality there. No metric code/tolerance was changed, and cosine/allclose diagnostics are not perceptual acceptance gates.

## Static checks and local commit handoff

The exact historical focused static-check argv was not preserved; these are runnable equivalents of the reported checks, not a claim of a new full `make lint` pass. Run after final document edits. Use the repository's configured pre-commit hooks and the same changed-file set, including untracked new files:

```bash
ft_worker env PRE_COMMIT_HOME=/workspace/tuning/ltx23_outpaint/precommit python -m pre_commit run --files \
  aitune/torch/jit/flextensor.py docs/guides/flextensor.md examples/README.md \
  examples/LTX23OutpaintAITuneFlexTensor/outpaint.py \
  examples/LTX23OutpaintAITuneFlexTensor/README.md \
  examples/LTX23OutpaintAITuneFlexTensor/pyproject.toml \
  tests/functional/pytorch/jit/021_jit_flextensor.py \
  tests/functional/pytorch/jit/022_jit_flextensor_fallback.py \
  tests/unit/torch/jit/test_flextensor.py plan.md report.md \
  work_logs/flextensor/validation_commands.md
ft_worker python tests/functional/scripts/validate.py
ft_worker python -m pytype aitune/torch/jit/flextensor.py -j auto
git diff --check
git diff --exit-code 6a63158 -- aitune ':!aitune/torch/jit/flextensor.py'
```

`make validate-functional` uses `uv run --script tests/functional/scripts/validate.py`; the direct Python invocation above uses dependencies already installed in the prepared runtime. The repository-wide lint target additionally runs `pre-commit run --all-files` and `pytype aitune tests -j auto`; the focused pytype result must not be described as that complete target. The pytype configuration disables `import-error` and `module-attr`. Retained outcomes are in `tuning/ltx23_outpaint/logs/final-precommit.log` and `final-functional-metadata.log`; later sweeps should have their own logs.

After reviewing/staging only intended changes, prepare the explicitly requested unsigned Conventional/DCO commit without changing signing configuration:

```bash
git commit --no-gpg-sign --signoff -m "fix(jit): isolate flextensor residency from compute tuning"
git log -1 --format='%H%n%G?%n%B'
```

Require signature status `N` and the correct `Signed-off-by` trailer before handoff. Only the user later signs/pushes; these are manual instructions, not agent actions:

```bash
git commit --amend --no-edit -s -S
git log -1 --show-signature
git push origin HEAD:kkleczewski/jit-flextensor
```
