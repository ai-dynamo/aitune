<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Evaluation handoff

Local runtime validation has been completed for the bundled smoke workload on one RTX 6000 Ada. Saved runs cover tuning,
Python correctness and benchmarking, inference, and both Original Model and AITune Dynamo correctness and benchmarking; see the
[evaluation records](results/README.md). Python and deployment correctness passed the provisional 0.70 hard
SSIM gate. AITune scores are below the current 0.95 soft threshold; the October 8 deployment reports record those warnings.

The remaining work includes representative quality/performance evaluation, a second GPU architecture, and
complete container build provenance. The local image ID and evaluation source snapshot are retained in the
[provenance record](../../results/flux-schnell/evaluation/20261008-tables/provenance.json); the locally built image
has no registry RepoDigest. Use the [setup, quickstart and deployment commands](README.md) for new runs.
The single-GPU workflow below also applies when reproducing or extending the existing local evaluation.

For the prepared **two-H100 context-parallel candidate**, use the exact
[multi-GPU evaluation and result-import commands](README.md#two-h100-evaluation).
That configuration launches two ranks on one node and covers Python tune, correctness, benchmark and inference;
it has no multi-GPU deployment targets and has not been run on H100. Return the documented
`flux-mgpu-measurements.tar.gz` archive, containing both ranks' reports, PNGs, raw measurements, artifact metadata,
topology and build provenance. Keep the large rank artifacts and caches on the evaluation machine.
The report command validates the evidence and fills only the marked multi-GPU README results block.
Its TensorRT timing-cache state differs from the historical cold-cache single-GPU runs and is recorded separately.

1. Build the container; retain its Docker image metadata/digest, build log and the frozen lockfile. Check the actual
   loaded PyTorch, CUDA, TensorRT and Dynamo versions in the generated environment reports.
2. Expose exactly one GPU and use the bundled inputs for a smoke evaluation. Run tune → correctness → benchmark → infer.
   A failure leaves logs in its unique run directory. Set a new artifact parent path before retrying tuning.
3. Replace the quickstart prompts with separate representative tuning, quality and performance manifests. Keep
   the same workload and seed distribution for both variants. Calibrate the SSIM threshold and add human/task quality
   evidence. Use a new artifact parent for this evaluation and retain the resolved configuration.
4. Run Original Model and AITune Dynamo services sequentially. Run deployment correctness and AIPerf for each;
   keep the same container, GPU, size, steps, seed, concurrency and measurement settings.
5. Repeat on a second GPU architecture, with a separate artifact and output directory. Never carry the artifact
   between architectures. Record the actual GPU models, VRAM, driver and any environment differences.
6. Inspect `compilation.json`, first-use/compiler logs, and raw AITune build events. Runtime compilation uses
   the union of PyTorch compiler-frame wall-time intervals, including tracing, lowering, and backend builds;
   zero means no new compiler-frame work was observed. This excludes prebuilt artifact loading and the original
   tuning/build phase. Keep first-use diagnostics separate because they can overlap compilation. Record TensorRT
   timing-cache state explicitly; the evaluated comparisons start with an empty cache in a fresh container.

Return these files from each evaluation:

- The copied recipe, representative manifests, container build metadata, and effective environment records.
- `tune/<run>/report.json`, `selection.json`, `aitune-tuning.json`, logs and retained compiler caches.
- Artifact provenance (`tuned.ait.json`) and the associated artifact, transferred through your approved artifact store.
- `correctness/<run>/report.json`, reference/tuned PNGs, and both worker reports.
- `benchmark/<run>/report.json`, worker reports, `timings.json`, `gpu-samples.jsonl`, and correctness evidence.
- Both `serve-<variant>/<run>/` records, including logs, service identity and raw server telemetry.
- Both deployment correctness reports and PNGs; both deployment benchmark reports and complete `aiperf/` directories.
- Quality threshold justification and application-level image review findings.

Retain correctness and performance reports for additional evaluations so the local result tables can be extended.
The current measurements cover the bundled prompts on one GPU architecture; they do not establish broader
model support or representative quality/performance.

## Proposed discovery entries (pending evidence)

The selected workspace has no AITune `examples/README.md` catalog or `docs/index.yml`. No ready-to-use catalog was
created here. Once evaluated in the intended public checkout, place this proposed row in the alphabetically sorted
text-to-image task group and adjust row spans. Only replace pending cells with observed support.

| Task | Model | Hub | Precision | Offloading | Single GPU | Multi-GPU | Deployment |
|---|---|---|---|---|---|---|---|
| Text-to-image | [FLUX.1-schnell](README.md) | [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | BF16 (pending) | - | Runtime compilation (pending) | - | Dynamo (pending) |

Catalog-relative Model link after integration: `flux-schnell/README.md`.
Proposed Model Recipes navigation target: `examples/flux-schnell/README.md`; adapt the entry to the destination's
existing YAML structure. Do not add the internal template, skill, or agent guide to public discovery pages.
