# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate tuned images against the original pipeline using the recipe's validation samples."""

import os
from logging import basicConfig, getLogger

import numpy as np
import torch.distributed as dist
from pydantic import BaseModel
from skimage.metrics import structural_similarity

from .checkpoint import check_checkpoint, load_checkpoint
from .config import ModelSettings, Recipe, Sample, ValidationSettings
from .distributed import initialize, is_rank_zero, shutdown
from .model import ImageSamples, generate_samples, get_pipeline

logger = getLogger(__name__)


class SampleValidationResult(BaseModel):
    id: str
    rgb_ssim: float
    max_abs_diff: float
    max_rel_error: float
    passed: bool
    warning: bool


class ValidationResult(ValidationSettings):
    samples: list[SampleValidationResult]
    passed: bool


class ValidationReport(BaseModel):
    recipe: str
    model: ModelSettings
    gpu_count: int
    validation_samples: list[Sample]
    validation: ValidationResult


def _validate(original: ImageSamples, tuned: ImageSamples, settings: ValidationSettings) -> ValidationResult:
    results: list[SampleValidationResult] = []
    for sample_id, baseline in original.items():
        candidate = tuned[sample_id]
        difference = np.abs(baseline - candidate)
        score = float(structural_similarity(baseline, candidate, data_range=1.0, channel_axis=-1))
        result = SampleValidationResult(
            id=sample_id,
            rgb_ssim=score,
            max_abs_diff=float(difference.max()),
            max_rel_error=float((difference / np.maximum(np.abs(baseline), 1e-8)).max()),
            passed=bool(np.isfinite(candidate).all() and score >= settings.threshold),
            warning=score < settings.soft_threshold,
        )
        if result.warning:
            logger.warning("%s RGB SSIM %.4f is below soft threshold %.4f", sample_id, score, settings.soft_threshold)
        results.append(result)
    return ValidationResult(**settings.model_dump(), samples=results, passed=all(result.passed for result in results))


def do_validation(recipe: Recipe) -> None:
    check_checkpoint(recipe)
    pipeline = get_pipeline(recipe)
    original = generate_samples(pipeline, recipe, "validation", "original")
    pipeline = load_checkpoint(pipeline, recipe)
    tuned = generate_samples(pipeline, recipe, "validation", "tuned")
    passed = True
    report_path = recipe.output_dir / "validation" / "report.json"
    if is_rank_zero():
        validation = _validate(original, tuned, recipe.validation)
        passed = validation.passed
        report = ValidationReport(
            recipe=str(recipe.path),
            model=recipe.model,
            gpu_count=recipe.gpu_count,
            validation_samples=recipe.samples("validation"),
            validation=validation,
        )
        report_path.write_text(report.model_dump_json(indent=2) + "\n")
        logger.info("Validation passed=%s; report=%s", passed, report_path)
    if dist.is_initialized():
        status = [passed]
        dist.broadcast_object_list(status, src=0)
        passed = status[0]
    if not passed:
        raise RuntimeError(f"Tuned output failed the recipe's RGB SSIM threshold; see {report_path}")


def run(recipe: Recipe) -> None:
    basicConfig(level=os.environ.get("AITUNE_LOG_LEVEL", "INFO"), force=True)
    initialize(recipe)
    try:
        do_validation(recipe)
    finally:
        shutdown()
