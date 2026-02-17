# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
