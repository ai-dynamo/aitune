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
"""Inference for E5 Large v2 embedding model."""

import logging

import torch
from aitune.torch import load
from aitune.torch.config import config as global_config

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

    # NOTE: workaround for SentenceTransformer to work with tuned model see below first dirty hack
    #       By default, we move modules to meta device after tuning to save memory. Our wrapper returns correct device for the module.
    #       However, SentenceTransformer overrides .device property, has one more reference to the same module we tuned, returns meta device by this additional reference.
    #       In encode() method, it calls .to(self.device) and tries to move modules that are already on right devices, breaking the inference.
    #       We need to move modules to cpu device after tuning to work with SentenceTransformer.
    global_config.device_after_tuning = "cpu"

    logger.info("Loading tuned model...")
    model = load(model, args.tuned_model_path)

    # XXX: Override .to method to do nothing.
    #      If we leave it as is and without specifying device in encode method, tensors will be moved to meta device, breaking the inference.
    # model.to = lambda *_args: None

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
