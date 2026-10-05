<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Evaluation handoff

Runtime validation has not been executed. Follow the exact [setup, quickstart and deployment commands](README.md)
on an available GPU. The authoring skill delegates validation to the engineer's infrastructure.

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
6. Inspect first-use/compiler logs and raw AITune build events. Pure compiler wall time is currently unavailable.
   Add an evaluated compiler-specific timing method before claiming all six metrics are measured; do not relabel
   search, conversion, first-use, or warmup as compilation. No zero compilation time is asserted.

Return these files from each evaluation:

- The copied recipe, representative manifests, container build metadata, and effective environment records.
- `tune/<run>/report.json`, `selection.json`, `aitune-tuning.json`, logs and retained compiler caches.
- Artifact provenance (`tuned.ait.json`) and the associated artifact, transferred through your approved artifact store.
- `correctness/<run>/report.json`, reference/tuned PNGs, and both worker reports.
- `benchmark/<run>/report.json`, worker reports, `timings.json`, `gpu-samples.jsonl`, and correctness evidence.
- Both `serve-<variant>/<run>/` records, including logs, service identity and raw server telemetry.
- Both deployment correctness reports and PNGs; both deployment benchmark reports and complete `aiperf/` directories.
- Quality threshold justification and application-level image review findings.

Please return the correctness and performance reports so hardware support and result tables can be established.
No model support or speedup is inferred from this code or the sample YAML.

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
