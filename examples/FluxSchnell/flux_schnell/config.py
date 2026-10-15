# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load recipe settings and sample files without command-line overrides."""

import argparse
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationInfo, field_validator, model_validator

SampleSplit = Literal["tune", "inference", "validation", "benchmark"]
SAMPLE_SPLITS: tuple[SampleSplit, ...] = ("tune", "inference", "validation", "benchmark")
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
Seed = Annotated[int, Field(strict=True, ge=0, lt=2**32)]
NonemptyStr = Annotated[str, Field(min_length=1)]


class GenerationInput(BaseModel):
    prompt: str
    seed: Seed

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt must not be empty")
        return value


class Sample(GenerationInput):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")


class ModelSettings(BaseModel):
    base_model: NonemptyStr
    revision: NonemptyStr


class ExecutionSettings(BaseModel):
    gpu_count: PositiveInt
    context_parallel: Literal["ulysses", "ring"]


class TuningSettings(BaseModel):
    quantization: StrictBool


class WorkloadSettings(BaseModel):
    width: PositiveInt = Field(multiple_of=16)
    height: PositiveInt = Field(multiple_of=16)
    num_inference_steps: int = Field(strict=True, ge=1, le=4)
    guidance_scale: float = Field(allow_inf_nan=False)
    max_sequence_length: int = Field(strict=True, ge=1, le=256)


class BenchmarkSettings(BaseModel):
    warmup_iterations: int = Field(strict=True, ge=0)
    repetitions: PositiveInt
    sampling_interval_s: float = Field(gt=0, allow_inf_nan=False)


class ValidationSettings(BaseModel):
    metric: Literal["rgb_ssim"]
    threshold: float = Field(ge=0, le=1)
    soft_threshold: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_thresholds(self) -> Self:
        if self.threshold > self.soft_threshold:
            raise ValueError("Validation threshold must not exceed soft_threshold")
        return self


class DeploymentSettings(BaseModel):
    port: int = Field(strict=True, gt=0, lt=65536)
    web_host: NonemptyStr
    web_port: int = Field(strict=True, gt=0, lt=65536)
    seed: Seed
    request_count: PositiveInt
    readiness_timeout_s: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_ports(self) -> Self:
        if self.port == self.web_port:
            raise ValueError("Deployment API and web ports must be distinct")
        return self


class InputSettings(BaseModel):
    tune: NonemptyStr
    validation: NonemptyStr
    inference: NonemptyStr
    benchmark: NonemptyStr


class Recipe(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Path = Field(exclude=True)
    name: NonemptyStr
    model: ModelSettings
    execution: ExecutionSettings
    tuning: TuningSettings
    artifact_path: Path
    output_dir: Path
    inputs: InputSettings
    workload: WorkloadSettings
    benchmark: BenchmarkSettings
    validation: ValidationSettings
    deployment: DeploymentSettings

    @field_validator("path")
    @classmethod
    def resolve_recipe_path(cls, value: Path) -> Path:
        return value.resolve()

    @field_validator("artifact_path", "output_dir", mode="before")
    @classmethod
    def validate_output_path(cls, value: object) -> object:
        if value == "":
            raise ValueError("Output paths must not be empty")
        return value

    @field_validator("artifact_path", "output_dir")
    @classmethod
    def resolve_output_path(cls, value: Path, info: ValidationInfo) -> Path:
        recipe_path = info.data.get("path")
        return (recipe_path.parent / value).resolve() if recipe_path is not None else value

    def resolve(self, value: str | Path) -> Path:
        return (self.path.parent / value).resolve()

    @property
    def gpu_count(self) -> int:
        return self.execution.gpu_count

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.deployment.port}"

    def samples(self, split: SampleSplit) -> list[Sample]:
        path = self.resolve(getattr(self.inputs, split))
        samples = [Sample.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
        if not samples:
            raise ValueError(f"{path}: sample file is empty")
        if len({sample.id for sample in samples}) != len(samples):
            raise ValueError(f"{path}: sample IDs must be unique")
        return samples


def argument_parser(description: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--recipe", type=Path, default=Path("recipes/single-gpu.yaml"), help="Recipe YAML path")
    return parser


def load_recipe(path: str | Path) -> Recipe:
    path = Path(path).resolve()
    with path.open() as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: recipe must be a YAML mapping")
    recipe = Recipe.model_validate({**data, "path": path})
    for split in SAMPLE_SPLITS:
        recipe.samples(split)
    return recipe
