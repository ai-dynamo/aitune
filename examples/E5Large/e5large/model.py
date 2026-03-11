# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model utilities for SentenceTransformer embedding."""

from sentence_transformers import SentenceTransformer


def get_model(model_name: str = "intfloat/e5-large-v2", device: str = "cuda"):
    """Get a pretrained SentenceTransformer.

    Args:
        model_name: SentenceTransformer model name or path
        device: Device to use for the model

    Returns:
        Embedding model
    """
    model = SentenceTransformer(model_name, device=device)
    model.to(device)
    model.eval()

    return model
