# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E5Large embedding backend — serves via dyn.dynamo_worker."""

import logging
import os
from pathlib import Path

import aitune.dynamo as dyn
import aitune.torch as ait
import numpy as np
import yaml

from ..model import MODEL_NAME, get_model


def _get_config() -> dict:
    with Path(os.environ.get("AITUNE_EXAMPLE_CONFIG_PATH", "config.yaml")).open() as f:
        return yaml.safe_load(f)


def main() -> None:
    """Main function."""
    logging.basicConfig(level=logging.INFO, force=True)
    cfg = _get_config()
    backend_cfg = cfg.get("Backend", {})
    model_name = backend_cfg.get("model_name", MODEL_NAME)
    tuned_model_path = backend_cfg.get("tuned_model_path")
    tuned_model_path = Path(tuned_model_path)
    if not (Path("checkpoints") / tuned_model_path).exists():
        raise RuntimeError(f"Tuned model not found at {tuned_model_path}. Run `tune` to tune the model.")

    model = get_model(model_name)

    # device_after_tuning="cpu" keeps AITune-wrapped sub-modules on CPU between requests;
    # encode(device="cuda") moves inputs to GPU for each inference pass.
    ait.config.device_after_tuning = "cpu"

    ait.load(model, tuned_model_path)

    def mapping(req) -> dict:
        sentences = req.input if isinstance(req.input, list) else [req.input]
        return {"sentences": sentences}

    def embed(sentences: list[str]) -> np.ndarray:
        return model.encode(
            sentences=sentences,
            batch_size=len(sentences),
            normalize_embeddings=True,
            show_progress_bar=False,
            device="cuda",
        )

    config = dyn.DynamoWorkerConfig(
        type="embedding",
        model_path=model_name,
        mapping=mapping,
    )
    dyn.dynamo_worker(embed, config)


if __name__ == "__main__":
    main()
