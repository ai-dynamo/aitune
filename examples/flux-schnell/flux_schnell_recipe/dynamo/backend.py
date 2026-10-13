# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamo's image protocol, backed by the same pipeline as Python commands."""

from __future__ import annotations

import io
import json
import os
import threading
import time
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import Config, InputRecord, Prompt, read_inputs
from ..records import artifact_metadata, contract, environment, source_identity, write_json

if TYPE_CHECKING:
    from diffusers import FluxPipeline
    from dynamo.common.protocols.image_protocol import NvCreateImageRequest

    from ..metrics import CompilationRecorder, Sampler


def map_request(request: NvCreateImageRequest, cfg: Config) -> Prompt:
    size = f"{cfg['workload']['width']}x{cfg['workload']['height']}"
    if request.size not in (None, size) or getattr(request, "n", 1) not in (None, 1):
        raise ValueError(f"This artifact accepts n=1 and size={size}")
    seed = request.nvext.seed if request.nvext is not None else None
    return {"prompt": request.prompt, "seed": cfg["deployment"]["seed"] if seed is None else seed}


def execute(args: Namespace, cfg: Config, directory: Path, rank: int) -> None:
    import aitune.dynamo as dyn

    from ..metrics import CompilationRecorder, Sampler, compilation_counters, prepare
    from ..model import generate, load_model, synchronize

    run_id = Path(args.run_dir).name
    name = f"flux-schnell-{args.variant}-{run_id}"
    lock = threading.Lock()
    sampler: Sampler | None = None
    # setup() initializes these before warmup() and serve().
    pipe: FluxPipeline
    identity: dict[str, Any]
    artifact: dict[str, Any]
    records: list[InputRecord]
    compilation: CompilationRecorder

    def setup() -> None:
        nonlocal pipe, identity, artifact, records, compilation
        identity = source_identity(args, cfg)
        artifact = artifact_metadata(cfg)
        if contract(cfg, identity) != artifact["contract"]:
            raise ValueError("Service source/configuration differs from the artifact")
        records = read_inputs(cfg["inputs"]["validation"]) + read_inputs(cfg["inputs"]["benchmark"])
        if any(r["seed"] != cfg["deployment"]["seed"] for r in records):
            raise ValueError("Validation/benchmark seeds must match the configured default service seed")
        compilation = CompilationRecorder(directory)
        pipe = load_model(args, cfg, args.variant, identity)

    def warmup() -> None:
        nonlocal sampler
        timing = prepare(pipe, cfg, records, compilation)
        sampler = Sampler(directory / "gpu-samples.jsonl", cfg["benchmark"]["sampling_interval_s"], stream=True).start()
        write_json(
            directory / "service.json",
            {
                "run_id": run_id,
                "model_name": name,
                "rank": rank,
                "variant": args.variant,
                "contract": contract(cfg, identity),
                "artifact_sha256": artifact["sha256"],
                "environment": environment(),
                "timing": timing,
                "directory": str(directory),
                "seed": cfg["deployment"]["seed"],
                "sampling_interval_s": cfg["benchmark"]["sampling_interval_s"],
            },
        )

    def serve(prompt: str, seed: int) -> bytes:
        import torch

        with lock:
            # Dynamo invokes the callable in an executor thread; CUDA's current device is thread-local.
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
            before = compilation_counters()
            compile_records = compilation.snapshot()
            start = time.time()
            image = generate(pipe, cfg, {"prompt": prompt, "seed": seed})
            synchronize()
            buffer = io.BytesIO()
            if rank == 0:
                image.save(buffer, format="PNG")
            compiled = compilation_counters() != before or compilation.snapshot() != compile_records
            with (directory / "requests.jsonl").open("a") as log:
                log.write(
                    json.dumps({"start_unix_s": start, "end_unix_s": time.time(), "compiled": compiled, "seed": seed})
                    + "\n"
                )
            if compiled:
                raise RuntimeError("Unexpected runtime compilation after service warmup; restart and inspect workload")
            return buffer.getvalue()

    try:
        dyn.dynamo_worker(
            serve,
            dyn.DynamoWorkerConfig(
                type="image",
                model_path=args.model_id or args.checkpoint,
                model_name=name,
                namespace="flux_" + run_id.replace("-", "_"),
                mapping=lambda request: map_request(request, cfg),
            ),
            setup=setup,
            warmup=warmup,
        )
    finally:
        if sampler:
            sampler.stop()
