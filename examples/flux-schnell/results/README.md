# Evaluation records

Local model workloads have been run on one NVIDIA RTX 6000 Ada Generation GPU (47.37 GiB reported VRAM).
As of 2026-10-08, tuning, Python correctness, Python benchmarking, inference, and both variants' Dynamo correctness and benchmarking
have saved results. The example remains a draft because the broader evaluation is incomplete.

The [two-H100 recipe](../recipes/multi-gpu.yaml) is prepared but unmeasured. Follow the
[multi-GPU procedure](../README.md#two-h100-evaluation) to generate `multi-gpu/` evidence beside the existing
`single-gpu/` runs. It includes separate rank artifacts, fresh-model correctness for both ranks/variants,
per-device telemetry, aggregate Python measurements and a command to import the results into the README.
No H100 measurements or multi-GPU deployment results are claimed here.

The records are in the repository-root `results/flux-schnell/single-gpu/` directory, mounted as
`/workspace/flux-schnell/results/single-gpu/` in the container. The links below point to those local records;
they require the retained results tree and are not bundled publication artifacts.

| Phase | Recorded evidence | Outcome |
|---|---|---|
| Tuning, October 1 | [Report](../../../results/flux-schnell/single-gpu/tune/20261001T151244Z-80180140/report.json) | Artifact saved; tuning/search took 617.75 s |
| Python correctness, October 2 | [Report](../../../results/flux-schnell/single-gpu/correctness/20261002T104828Z-46b89882/report.json) | All four samples passed SSIM ≥ 0.70; scores 0.724–0.938 |
| Python benchmark, October 2 | [Report](../../../results/flux-schnell/single-gpu/benchmark/20261002T105729Z-d819c457/report.json) | Original Model: 0.2985 images/s, 3350.31 ms mean latency; AITune: 0.3914 images/s, 2555.12 ms |
| Inference, October 2 | [Report](../../../results/flux-schnell/single-gpu/infer/20261002T115010Z-754fe2a0/aitune/report.json) | Tuned artifact generated [lighthouse.png](../../../results/flux-schnell/single-gpu/infer/20261002T115010Z-754fe2a0/aitune/images/lighthouse.png) |
| AITune Dynamo serving, October 7 | [Service record](../../../results/flux-schnell/single-gpu/serve-aitune/20261007T094838Z-791f2df4/service.json) | Saved artifact served through the image endpoint |
| AITune Dynamo correctness, October 7 | [Report](../../../results/flux-schnell/single-gpu/deployment-correctness-aitune/20261007T100243Z-49c24c86/report.json) | All four samples passed SSIM ≥ 0.70; scores 0.724–0.938 |
| Python benchmark, October 8 | [Report](../../../results/flux-schnell/single-gpu/benchmark/20261008T135335Z-49ad1389/report.json) | Both fresh workers passed correctness; Original Model: 0.2981 images/s, 3355.09 ms; AITune: 0.3867 images/s, 2585.88 ms |
| Original Model Dynamo benchmark, October 8 | [Report](../../../results/flux-schnell/single-gpu/deployment-benchmark-original/20261008T114920Z-05e1b0a5/report.json) | Fresh correctness passed; 0.3039 images/s, 3288.71 ms mean latency over 20 requests |
| AITune Dynamo benchmark, October 8 | [Report](../../../results/flux-schnell/single-gpu/deployment-benchmark-aitune/20261008T120503Z-a0415d82/report.json) | Fresh correctness passed with soft-threshold warnings; 0.3764 images/s, 2654.99 ms over 20 requests |

These runs used BF16, 1024×1024 images, four denoising steps, seed 42, and batch/concurrency 1. The Python
benchmark measured 20 images per variant using the four bundled validation prompts. See the
[model page](../README.md#recipe-and-performance) for the six-metric tables and measurement definitions.
Earlier correctness runs failed the initial 0.95 hard gate. The October 2 and 7 passing records predate the optional
`soft_threshold` field; their scores would all trigger warnings under the current 0.95 soft threshold.
Historical reports retain their original acceptance criteria.

The October 8 Python and Dynamo reports include all six metrics. Compiler-frame timings record 0.00 s of
new compilation for Original Model, 311.41 s for Python AITune, and 316.98 s for Dynamo AITune;
these exclude the original AOT artifact builds. The Python comparison reused verified references and ran
each benchmark worker's fresh-model correctness check; its
[exact invocation](../../../results/flux-schnell/evaluation/20261008-tables/python-command.json) is retained.
Raw AIPerf exports, GPU samples, server request records, compilation intervals, and correctness images are retained.
The initial AIPerf attempt rejected `--warmup-request-count 0`; its failed run is preserved, and the successful
runs use the corrected command that omits that option.

Evaluation on a second GPU architecture and representative quality/performance inputs remain outstanding.
The [provenance record](../../../results/flux-schnell/evaluation/20261008-tables/provenance.json) retains the local
image ID, source snapshot/hashes, and cache state. Saved environment reports include library and driver versions
but leave the registry digest null; complete image build provenance remains outstanding.

Commands create timestamped run directories below `single-gpu/` in the results mount. Keep the complete tree, including
the artifact, its JSON provenance, raw AITune build records, caches, reference PNGs, resolved configurations,
per-variant environments, latency records, NVML samples, and raw AIPerf artifacts.

Do not treat the bundled prompt samples as a representative quality or performance suite. Before publishing,
evaluate a documented workload on at least two GPU architectures and return the reports listed in
[the evaluation handoff](../EVALUATION.md). Missing measurements remain null with reasons.
