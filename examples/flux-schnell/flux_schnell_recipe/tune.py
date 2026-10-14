# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect, wrap, tune, and save FLUX.1-schnell on one or two GPUs."""

from __future__ import annotations

import logging
import time
from argparse import Namespace
from typing import TYPE_CHECKING, Any

from .config import Config, argument_parser, configure, configure_cache, rank_config, read_inputs, write_json
from .distributed import collect, initialize, shutdown
from .model import contract, load_model, synchronize

if TYPE_CHECKING:
    from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
    from torch import Tensor

logger = logging.getLogger(__name__)


def tune_model(args: Namespace, cfg: Config) -> dict[str, Any]:
    if cfg.artifact.exists():
        raise FileExistsError(f"Choose a new artifact_path to retune: {cfg.artifact}")
    cfg.artifact.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Loading PyTorch and AITune for tuning")
    import torch

    import aitune.torch as ait
    from aitune.torch.backend import (
        TorchInductorJitBackend,
        TorchInductorJitBackendConfig,
        TorchTensorRTConfig,
        TorchTensorRTJitBackend,
        TorchTensorRTJitBackendConfig,
    )

    logger.info("Loading base model for tuning")
    pipe = load_model(args, cfg)
    samples = read_inputs(cfg["inputs"]["tune"])
    # AITune batches strings into lists and scalar seeds into one-element tensors.

    def invoke(prompt: list[str], seed: Tensor) -> FluxPipelineOutput:
        return pipe(
            prompt=prompt,
            **cfg["workload"],
            output_type="pil",
            generator=torch.Generator("cpu").manual_seed(int(seed.item())),
        )

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
        compile_options["timing_cache_path"] = str(cfg.artifact.parent / "tensorrt-timing-cache.bin")
    strategy = ait.MaxThroughputStrategy(
        backends=[
            TorchTensorRTJitBackend(
                config=TorchTensorRTJitBackendConfig(
                    dynamic=False, compile_config=TorchTensorRTConfig(**compile_options)
                )
            ),
            TorchInductorJitBackend(config=TorchInductorJitBackendConfig(dynamic=False)),
        ]
    ).enable_find_max_batch_size(False)
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
    logger.info("Tuning completed in %.1fs; saving artifact to %s", tuning_s, cfg.artifact)
    ait.save(pipe, cfg.artifact)
    metadata = {
        "contract": contract(args, cfg),
        "inspection_s": inspection_s,
        "tuning_search_s": tuning_s,
        "artifact": str(cfg.artifact),
    }
    write_json(str(cfg.artifact) + ".json", metadata)
    return metadata


def main() -> None:
    args = argument_parser(__doc__, source=True).parse_args()
    cfg = configure(args)
    configure_cache(cfg)
    try:
        rank = initialize(cfg)
        ranks = collect(tune_model(args, rank_config(cfg, rank)))
        if rank == 0:
            report = {"gpu_count": len(ranks), "ranks": ranks}
            if len(ranks) > 1:
                report["artifacts"] = [
                    {"rank": rank, "file": rank_config(cfg, rank).artifact.name} for rank in range(len(ranks))
                ]
                write_json(cfg.artifact.parent / "manifest.json", report)
            write_json(cfg.output / "tune" / "report.json", report)
            print(cfg.output / "tune" / "report.json", flush=True)
    finally:
        shutdown()


if __name__ == "__main__":
    main()
