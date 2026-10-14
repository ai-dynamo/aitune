# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve the same FLUX pipeline through Dynamo's image protocol."""

from __future__ import annotations

import io
import json
import os
import threading
import time
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import Config, Prompt, argument_parser, configure, configure_cache, rank_config, read_inputs, write_json
from ..distributed import collect, initialize, rank_directory, shutdown
from ..metrics import CompilationRecorder, Sampler, compilation_counters, interval_seconds, phase_duration, prepare
from ..model import contract, generate, load_model, load_tuned, synchronize

if TYPE_CHECKING:
    from diffusers import FluxPipeline
    from dynamo.common.protocols.image_protocol import NvCreateImageRequest


class FluxDynamoBackend:
    pipe: FluxPipeline
    compilation: CompilationRecorder

    def __init__(self, args: Namespace, cfg: Config, rank: int) -> None:
        self.args, self.cfg, self.rank = args, cfg, rank
        self.output = cfg.output / "serve" / args.variant
        self.directory = rank_directory(cfg, self.output, rank)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.name = f"flux-schnell-{args.variant}-{cfg['name']}-{os.environ.get('FLUX_RUN_ID', 'local')}"
        self.lock = threading.Lock()
        self.sampler: Sampler | None = None

    def setup(self) -> None:
        self.compilation = CompilationRecorder(self.directory)
        self.pipe = load_model(self.args, self.cfg)
        if self.args.variant == "aitune":
            self.pipe = load_tuned(self.pipe, self.args, self.cfg)

    def warmup(self) -> None:
        records = read_inputs(self.cfg["inputs"]["validation"]) + read_inputs(self.cfg["inputs"]["benchmark"])
        timing = prepare(self.pipe, self.cfg, records, self.compilation)
        (self.directory / "requests.jsonl").write_text("")
        self.sampler = Sampler(
            self.directory / "gpu-samples.jsonl", self.cfg["benchmark"]["sampling_interval_s"], stream=True
        ).start()
        ranks = collect({
            "rank": self.rank,
            "directory": str(self.directory),
            "gpu_uuid": self.sampler.uuid,
            "sampling_interval_s": self.cfg["benchmark"]["sampling_interval_s"],
            "timing": timing,
        })
        compiler_records = collect(self.compilation.snapshot())
        if self.rank == 0:
            timings = [rank["timing"] for rank in ranks]
            timing.update({
                "first_use_s": phase_duration(timings, "first_use"),
                "warmup_s": phase_duration(timings, "warmup"),
                "compilation_s": interval_seconds([row for rows in compiler_records for row in rows]),
                "raw_compilation": [row["raw_compilation"] for row in timings],
            })
            write_json(
                self.output / "service.json",
                {
                    "model_name": self.name,
                    "variant": self.args.variant,
                    "contract": contract(self.args, self.cfg),
                    "seed": self.cfg["deployment"]["seed"],
                    "timing": timing,
                    "ranks": ranks,
                },
            )

    def map_request(self, request: NvCreateImageRequest) -> Prompt:
        size = f"{self.cfg['workload']['width']}x{self.cfg['workload']['height']}"
        if request.size not in (None, size) or getattr(request, "n", 1) not in (None, 1):
            raise ValueError(f"This artifact accepts n=1 and size={size}")
        seed = request.nvext.seed if request.nvext is not None else None
        return {"prompt": request.prompt, "seed": self.cfg["deployment"]["seed"] if seed is None else seed}

    def generate(self, prompt: str, seed: int) -> bytes:
        import torch

        with self.lock:
            # Dynamo calls this in an executor thread; the CUDA device is thread-local.
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
            before, records = compilation_counters(), self.compilation.snapshot()
            start = time.time()
            image = generate(self.pipe, self.cfg, {"prompt": prompt, "seed": seed})
            synchronize()
            buffer = io.BytesIO()
            if self.rank == 0:
                image.save(buffer, format="PNG")
            compiled = compilation_counters() != before or self.compilation.snapshot() != records
            with (self.directory / "requests.jsonl").open("a") as log:
                log.write(
                    json.dumps({
                        "start_unix_s": start,
                        "end_unix_s": time.time(),
                        "compiled": compiled,
                        "seed": seed,
                    })
                    + "\n"
                )
            if any(collect(compiled)):
                raise RuntimeError("Unexpected compilation after service warmup; restart and inspect the workload")
            return buffer.getvalue()

    def run(self) -> None:
        import aitune.dynamo as dyn

        try:
            dyn.dynamo_worker(
                self.generate,
                dyn.DynamoWorkerConfig(
                    type="image",
                    model_path=self.args.model_id or self.args.checkpoint,
                    model_name=self.name,
                    namespace=self.name.replace("-", "_"),
                    mapping=self.map_request,
                ),
                setup=self.setup,
                warmup=self.warmup,
            )
        finally:
            if self.sampler:
                self.sampler.stop()


def main() -> None:
    parser = argument_parser(__doc__, source=True)
    parser.add_argument("--variant", choices=("original", "aitune"), default="aitune")
    args = parser.parse_args()
    cfg = configure(args)
    configure_cache(cfg)
    if int(os.environ.get("RANK", "0")) == 0 and (pid_file := os.environ.get("FLUX_RANK0_PID_FILE")):
        Path(pid_file).write_text(str(os.getpid()))
    try:
        rank = initialize(cfg)
        FluxDynamoBackend(args, rank_config(cfg, rank), rank).run()
    finally:
        shutdown()


if __name__ == "__main__":
    main()
