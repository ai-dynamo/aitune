# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect, tune, and persist the pipeline and its provenance."""

import logging
import time

from .data import read_inputs
from .model import load_model, synchronize
from .records import configure_cache, contract, environment, file_sha, new_run, read_json, source_identity, write_json

logger = logging.getLogger(__name__)


def run(args, cfg, run_dir=None):
    # Exclusive directory creation also prevents two tuning processes overwriting one artifact.
    if run_dir is None:
        cfg.artifact.parent.mkdir(parents=True, exist_ok=False)
        run_dir = new_run(cfg, "tune")
    configure_cache(run_dir)
    logger.info("Loading PyTorch and AITune for tuning")
    import torch
    import aitune.torch as ait
    from aitune.torch.backend import (TorchInductorJitBackend, TorchInductorJitBackendConfig,
                                     TorchTensorRTJitBackend, TorchTensorRTJitBackendConfig, TorchTensorRTConfig)
    source = source_identity(args, cfg)
    logger.info("Loading base model for tuning")
    pipe = load_model(args, cfg, source=source)
    samples = read_inputs(cfg["inputs"]["tune"])
    # AITune batches strings into lists and scalar seeds into one-element tensors.
    def invoke(prompt, seed):
        return pipe(prompt=prompt, **cfg["workload"], output_type="pil",
                    generator=torch.Generator("cpu").manual_seed(int(seed.item())))

    dataset = [{"prompt": item["prompt"], "seed": torch.tensor(item["seed"])} for item in samples]
    logger.info("Inspecting pipeline with %d tuning inputs; synchronizing GPUs", len(samples))
    synchronize()
    start = time.perf_counter()
    info = ait.inspect(pipe, dataset, inference_function=invoke, number_of_iterations=1, warmup_iterations=2)
    info.describe()
    inspection_s = time.perf_counter() - start
    logger.info("Pipeline inspection complete in %.1fs; configuring tuning candidates", inspection_s)
    compile_options = {"min_block_size": 50, "truncate_double": True}
    if cfg["execution"]["gpu_count"] > 1:
        compile_options["use_distributed_mode_trace"] = True
        # Both ranks must describe the same strategy. AITune adds rank suffixes
        # to this common base path when it opens the actual timing cache files.
        compile_options["timing_cache_path"] = str(run_dir.parent / "tensorrt-timing-cache.bin")
    strategy = ait.MaxThroughputStrategy(backends=[
        TorchTensorRTJitBackend(config=TorchTensorRTJitBackendConfig(
            dynamic=False, compile_config=TorchTensorRTConfig(**compile_options))),
        TorchInductorJitBackend(config=TorchInductorJitBackendConfig(dynamic=False)),
    ]).enable_find_max_batch_size(False)
    others = [m for m in info.get_modules() if m.name != "transformer"]
    pipe = ait.wrap(pipe, others, strategy=ait.MaxThroughputStrategy.for_aot().enable_find_max_batch_size(False))
    pipe.transformer = ait.module.Module(pipe.transformer, name="transformer", strategy=strategy)
    logger.info("Starting AITune candidate compilation and search; this can take several minutes")
    synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        ait.tune(invoke, dataset, batch_sizes=[1], ignore_failing_modules=False)
    synchronize()
    tuning_s = time.perf_counter() - start
    tuning_end = time.perf_counter()
    logger.info("Tuning completed in %.1fs; saving artifact to %s", tuning_s, cfg.artifact)
    ait.save(pipe, cfg.artifact)
    report = read_json(run_dir / "aitune-tuning.json")
    selection = [{"component": m["module_name"], "graphs": [
        {"graph": g["graph_name"], "selected_backend": g.get("selected_backend"),
         "strategy_config": g.get("strategy_config"), "input_spec": g["input_spec"]}
        for g in m["graphs"]]} for m in report["modules"]]
    write_json(run_dir / "selection.json", selection)
    metadata = {"contract": contract(cfg, source), "sha256": file_sha(cfg.artifact),
                "tune_run": str(run_dir), "environment": environment(),
                "inspection_s": inspection_s, "tuning_search_s": tuning_s,
                "tuning_start_monotonic_s": start, "tuning_end_monotonic_s": tuning_end,
                "compilation_s": None,
                "compilation_missing_reason": "Backend build timers include conversion and validation; pure compilation is not isolated",
                "raw_build_timings": str(run_dir / "aitune-tuning.json"),
                "cache_state": "new empty per-run compiler caches", "selection": selection,
                "tuning_inputs_sha256": file_sha(cfg["inputs"]["tune"])}
    write_json(str(cfg.artifact) + ".json", metadata)
    write_json(run_dir / "report.json", metadata)
    logger.info("Saved tuning report: %s", run_dir / "report.json")
    print(run_dir / "report.json")
