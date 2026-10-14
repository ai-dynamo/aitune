# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load recipe settings and sample files without command-line overrides."""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, overload

import yaml

SampleSplit = Literal["tune", "inference", "validation", "benchmark"]
SAMPLE_SPLITS: tuple[SampleSplit, ...] = ("tune", "inference", "validation", "benchmark")
RecipeKey = Literal[
    "name",
    "model",
    "execution",
    "tuning",
    "artifact_path",
    "output_dir",
    "inputs",
    "workload",
    "benchmark",
    "validation",
    "deployment",
]


class GenerationInput(TypedDict):
    prompt: str
    seed: int


class Sample(GenerationInput):
    id: str


class ModelSettings(TypedDict):
    base_model: str
    revision: str


class ExecutionSettings(TypedDict):
    gpu_count: int
    context_parallel: Literal["ulysses", "ring"]


class TuningSettings(TypedDict):
    quantization: bool


class WorkloadSettings(TypedDict):
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    max_sequence_length: int


class BenchmarkSettings(TypedDict):
    warmup_iterations: int
    repetitions: int
    sampling_interval_s: float


class ValidationSettings(TypedDict):
    metric: Literal["rgb_ssim"]
    threshold: float
    soft_threshold: float


class DeploymentSettings(TypedDict):
    port: int
    web_host: str
    web_port: int
    seed: int
    request_count: int
    readiness_timeout_s: float


class RecipeData(TypedDict):
    name: str
    model: ModelSettings
    execution: ExecutionSettings
    tuning: TuningSettings
    artifact_path: str
    output_dir: str
    inputs: dict[SampleSplit, str]
    workload: WorkloadSettings
    benchmark: BenchmarkSettings
    validation: ValidationSettings
    deployment: DeploymentSettings


@dataclass(frozen=True)
class Recipe:
    path: Path
    data: RecipeData

    @overload
    def __getitem__(self, key: Literal["name", "artifact_path", "output_dir"]) -> str: ...

    @overload
    def __getitem__(self, key: Literal["model"]) -> ModelSettings: ...

    @overload
    def __getitem__(self, key: Literal["execution"]) -> ExecutionSettings: ...

    @overload
    def __getitem__(self, key: Literal["tuning"]) -> TuningSettings: ...

    @overload
    def __getitem__(self, key: Literal["inputs"]) -> dict[SampleSplit, str]: ...

    @overload
    def __getitem__(self, key: Literal["workload"]) -> WorkloadSettings: ...

    @overload
    def __getitem__(self, key: Literal["benchmark"]) -> BenchmarkSettings: ...

    @overload
    def __getitem__(self, key: Literal["validation"]) -> ValidationSettings: ...

    @overload
    def __getitem__(self, key: Literal["deployment"]) -> DeploymentSettings: ...

    def __getitem__(self, key: RecipeKey) -> object:
        return self.data[key]

    def resolve(self, value: str) -> Path:
        return (self.path.parent / value).resolve()

    @property
    def artifact_path(self) -> Path:
        return self.resolve(self["artifact_path"])

    @property
    def output_dir(self) -> Path:
        return self.resolve(self["output_dir"])

    @property
    def gpu_count(self) -> int:
        return self["execution"]["gpu_count"]

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self['deployment']['port']}"

    def samples(self, split: SampleSplit) -> list[Sample]:
        path = self.resolve(self["inputs"][split])
        samples: list[Sample] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        seen: set[str] = set()
        for sample in samples:
            sample_id = sample["id"]
            if not isinstance(sample_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id) or sample_id in seen:
                raise ValueError(f"{path}: sample IDs must be unique, safe filenames")
            if not isinstance(sample["prompt"], str) or not sample["prompt"].strip():
                raise ValueError(f"{path}: {sample_id} needs a nonempty prompt")
            if type(sample["seed"]) is not int or not 0 <= sample["seed"] < 2**32:
                raise ValueError(f"{path}: {sample_id} needs an integer seed in [0, 2**32)")
            seen.add(sample_id)
        if not samples:
            raise ValueError(f"{path}: sample file is empty")
        return samples


def argument_parser(description: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--recipe", type=Path, default=Path("recipes/single-gpu.yaml"), help="Recipe YAML path")
    return parser


def load_recipe(path: str | Path) -> Recipe:
    path = Path(path).resolve()
    with path.open() as stream:
        data = yaml.safe_load(stream)
    required = {
        "model": ("base_model", "revision"),
        "execution": ("gpu_count", "context_parallel"),
        "tuning": ("quantization",),
        "inputs": SAMPLE_SPLITS,
        "workload": ("width", "height", "num_inference_steps", "guidance_scale", "max_sequence_length"),
        "benchmark": ("warmup_iterations", "repetitions", "sampling_interval_s"),
        "validation": ("metric", "threshold", "soft_threshold"),
        "deployment": ("port", "seed", "request_count", "readiness_timeout_s", "web_host", "web_port"),
    }
    for key in ("name", "artifact_path", "output_dir"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"{path}: missing or invalid {key}")
    for section, keys in required.items():
        if not isinstance(data.get(section), dict) or set(keys) - data[section].keys():
            raise ValueError(f"{path}: {section} requires {', '.join(keys)}")
    recipe = Recipe(path, data)
    _validate_settings(recipe)
    for split in SAMPLE_SPLITS:
        recipe.samples(split)
    return recipe


def _validate_settings(recipe: Recipe) -> None:
    workload = recipe["workload"]
    benchmark = recipe["benchmark"]
    validation = recipe["validation"]
    deployment = recipe["deployment"]
    positive = [
        recipe.gpu_count,
        workload["width"],
        workload["height"],
        benchmark["repetitions"],
        deployment["request_count"],
    ]
    if any(type(value) is not int or value < 1 for value in positive):
        raise ValueError("GPU count, dimensions, repetitions, and request count must be positive integers")
    if workload["width"] % 16 or workload["height"] % 16:
        raise ValueError("FLUX image dimensions must be multiples of 16")
    if not 1 <= workload["num_inference_steps"] <= 4 or not 1 <= workload["max_sequence_length"] <= 256:
        raise ValueError("Schnell requires 1–4 steps and a maximum sequence length of 1–256")
    if recipe["execution"]["context_parallel"] not in ("ulysses", "ring"):
        raise ValueError("execution.context_parallel must be ulysses or ring")
    if type(recipe["tuning"]["quantization"]) is not bool:
        raise ValueError("tuning.quantization must be a boolean")
    if benchmark["warmup_iterations"] < 0 or benchmark["sampling_interval_s"] <= 0:
        raise ValueError("Benchmark warmup must be nonnegative and sampling interval positive")
    if validation["metric"] != "rgb_ssim" or not 0 <= validation["threshold"] <= validation["soft_threshold"] <= 1:
        raise ValueError("Validation requires rgb_ssim and 0 <= threshold <= soft_threshold <= 1")
    if (
        not all(0 < deployment[key] < 65536 for key in ("port", "web_port"))
        or deployment["port"] == deployment["web_port"]
    ):
        raise ValueError("Deployment API and web ports must be distinct valid port numbers")
    if deployment["readiness_timeout_s"] <= 0 or not 0 <= deployment["seed"] < 2**32:
        raise ValueError("Deployment timeout must be positive and seed in [0, 2**32)")
