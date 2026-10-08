# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit, bounded single-GPU and context-parallel Schnell configurations."""

from dataclasses import dataclass
from pathlib import Path
import re

import yaml


@dataclass(frozen=True)
class Config:
    path: Path
    data: dict

    def __getitem__(self, key):
        return self.data[key]

    @property
    def output(self):
        return Path(self["output_dir"])

    @property
    def artifact(self):
        return Path(self["artifact_path"])


def load_config(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Recipe must be a YAML mapping")
    for key in ("model", "environment", "hardware", "execution", "precision", "deployment",
                "inputs", "workload", "tuning", "benchmark", "validation"):
        if not isinstance(raw.get(key), dict):
            raise ValueError(f"{key} must be a mapping")
    if not isinstance(raw.get("name"), str) or not re.fullmatch(r"[a-z0-9-]+", raw["name"]):
        raise ValueError("name must contain lowercase letters, numbers, or hyphens")
    execution = raw["execution"]
    count = execution.get("gpu_count")
    if type(count) is not int or count not in (1, 2):
        raise ValueError("Implemented layouts: one GPU or two GPUs on one node")
    if execution != {"gpu_count": count, "parallelism": "none" if count == 1 else "context-ulysses",
                     "compilation": "runtime", "weight_offloading": False}:
        raise ValueError("Use runtime compilation without offloading; two GPUs require context-ulysses")
    if raw["precision"] != {"allowed": ["bf16"]}:
        raise ValueError("This recipe supports BF16 without quantization")
    if raw["tuning"] != {"batch_sizes": [1]}:
        raise ValueError("Only batch size 1 is implemented")
    if raw["model"].get("base_model") != "black-forest-labs/FLUX.1-schnell":
        raise ValueError("The base architecture must be FLUX.1-schnell")
    if not re.fullmatch(r"[0-9a-f]{40}", str(raw["model"].get("revision", ""))):
        raise ValueError("model.revision must pin a 40-character hub commit")
    if raw["model"].get("checkpoint_format") != "diffusers-directory":
        raise ValueError("Only complete Diffusers checkpoint directories are accepted")
    w = raw["workload"]
    if set(w) != {"width", "height", "num_inference_steps", "guidance_scale", "max_sequence_length"}:
        raise ValueError("workload must specify width, height, steps, guidance, and sequence length")
    for key in ("width", "height"):
        if type(w[key]) is not int or not 256 <= w[key] <= 2048 or w[key] % 16:
            raise ValueError(f"{key} must be a multiple of 16 between 256 and 2048; re-tune after changes")
    if type(w["num_inference_steps"]) is not int or not 1 <= w["num_inference_steps"] <= 4:
        raise ValueError("Schnell recipes use 1–4 denoising steps")
    if w["guidance_scale"] != 0:
        raise ValueError("Schnell requires guidance_scale: 0")
    if type(w["max_sequence_length"]) is not int or not 1 <= w["max_sequence_length"] <= 256:
        raise ValueError("Schnell max_sequence_length must be between 1 and 256")
    if count > 1 and (w["max_sequence_length"] % count or (w["width"] // 16 * (w["height"] // 16)) % count):
        raise ValueError("Context parallelism requires text and packed image sequence lengths divisible by GPU count")
    b = raw["benchmark"]
    for key in ("warmup_iterations", "repetitions"):
        if type(b.get(key)) is not int or b[key] < 1:
            raise ValueError(f"benchmark.{key} must be positive")
    if not isinstance(b.get("sampling_interval_s"), (int, float)) or not 0.01 <= b["sampling_interval_s"] <= 1:
        raise ValueError("sampling_interval_s must be between 0.01 and 1")
    v = raw["validation"]
    if v.get("metric") != "rgb_ssim" or type(v.get("threshold")) not in (int, float) or not 0 < v["threshold"] <= 1:
        raise ValueError("validation must specify rgb_ssim and threshold in (0,1]")
    soft_threshold = v.get("soft_threshold")
    if soft_threshold is not None and (type(soft_threshold) not in (int, float)
            or not v["threshold"] <= soft_threshold <= 1):
        raise ValueError("validation.soft_threshold must be between validation.threshold and 1, or null")
    d = raw["deployment"]
    if count > 1 and d != {"targets": []}:
        raise ValueError("The multi-GPU recipe implements Python phases only; set deployment.targets: []")
    if count == 1 and (d.get("targets") != ["dynamo"] or d.get("concurrency") != 1):
        raise ValueError("Deployment implements Dynamo at concurrency 1")
    for key in (("port", "request_count", "readiness_timeout_s") if count == 1 else ()):
        if type(d.get(key)) is not int or d[key] < 1:
            raise ValueError(f"deployment.{key} must be positive")
    if count == 1 and (d["port"] > 65535 or type(d.get("seed")) is not int or not 0 <= d["seed"] < 2**32):
        raise ValueError("Invalid deployment port or seed")

    def resolved(value):
        if not isinstance(value, str) or not value:
            raise ValueError("Paths must be nonempty strings")
        return str((path.parent / value).resolve())

    for key in ("artifact_path", "output_dir"):
        raw[key] = resolved(raw.get(key))
    for phase in ("tune", "validation", "inference", "benchmark"):
        raw["inputs"][phase] = resolved(raw["inputs"].get(phase))
    raw["environment"]["dockerfile"] = resolved(raw["environment"]["dockerfile"])
    return Config(path, raw)
