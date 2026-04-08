# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FLUX image backend — serves via dyn.dynamo_worker."""

import io
import os
from pathlib import Path

import yaml

import aitune.dynamo as dyn
import aitune.torch as ait

from ..model import MODEL_NAME, get_pipeline

_DEFAULT_STEPS = 20
_DEFAULT_GUIDANCE = 7.5
_DEFAULT_MAX_SEQ_LEN = 77


def _get_config() -> dict:
    with Path(os.environ.get("AITUNE_EXAMPLE_CONFIG_PATH", "config.yaml")).open() as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = _get_config()
    backend_cfg = cfg.get("Backend", {})
    model_name = backend_cfg.get("model_name", MODEL_NAME)
    tuned_model_path = backend_cfg.get("tuned_model_path")

    if not Path("checkpoints", tuned_model_path).exists():
        raise RuntimeError(f"Tuned model not found at {tuned_model_path}")

    pipeline = get_pipeline(model_name)
    ait.load(pipeline, tuned_model_path)

    def mapping(req) -> dict:
        w, h = (int(x) for x in (req.size or "1024x1024").split("x"))
        return {
            "prompt": req.prompt,
            "height": h,
            "width": w,
            "num_inference_steps": _DEFAULT_STEPS,
            "guidance_scale": _DEFAULT_GUIDANCE,
            "max_sequence_length": _DEFAULT_MAX_SEQ_LEN,
        }

    def generate(
        prompt: str, height: int, width: int, num_inference_steps: int, guidance_scale: float, max_sequence_length: int
    ) -> bytes:
        result = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
            return_dict=True,
        )
        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")
        return buf.getvalue()

    config = dyn.DynamoWorkerConfig(
        type="image",
        model_path=model_name,
        mapping=mapping,
    )
    dyn.dynamo_worker(generate, config)


if __name__ == "__main__":
    main()
