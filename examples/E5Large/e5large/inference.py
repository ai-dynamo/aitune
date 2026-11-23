# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inference for E5 Large v2 embedding model."""

import logging

import torch
from aitune.torch import load
from aitune.utils.monitoring import annotate
from aitune_examples_common.checkpoint import relocated_checkpoint_path

from .cmd_args import get_parser
from .model import get_model

logging.basicConfig(level=logging.DEBUG, force=True)

logger = logging.getLogger(__name__)


def main():
    """Main function."""
    args = get_parser().parse_args()

    sentence = args.prompt

    logger.info("Getting model...")
    model = get_model(model_name=args.model_name, device="cuda")

    model.encode(sentences=["query: test"], batch_size=1, convert_to_tensor=True)

    logger.info("Loading tuned model...")
    model = load(model, relocated_checkpoint_path(args.tuned_model_path))

    @annotate(name="inference", color="green")
    def infer(*args, **kwargs):
        return model.encode(
            *args,
            **kwargs,
            convert_to_tensor=True,
            convert_to_numpy=False,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=1,
            device="cuda",
        )

    embeddings = infer(sentences=[sentence, sentence, sentence, sentence])
    logger.info("Embeddings: %s", embeddings.tolist())  # noqa: T201


if __name__ == "__main__":
    with torch.no_grad():
        main()
