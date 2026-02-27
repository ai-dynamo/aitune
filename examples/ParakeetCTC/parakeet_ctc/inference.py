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

"""Inference for ASR model."""

import os
from logging import basicConfig, getLogger
from pathlib import Path

import torch
from nemo.collections.asr.parts.mixins.transcription import InternalTranscribeConfig, TranscribeConfig

from aitune.torch import load

from .model import get_model
from .tune import parse_args

logger = getLogger(__name__)


def do_inference(
    model_name: str,
    audio_path: Path,
    tuned_model_path: Path,
):
    """Do inference on a tuned ParakeetCTC model."""
    model = get_model(model_name=model_name)
    pipeline = load(model, tuned_model_path)

    def infer(*args, **kwargs):
        return pipeline.transcribe(
            *args,
            **kwargs,
            override_config=TranscribeConfig(
                batch_size=kwargs["batch_size"],
                verbose=False,
                _internal=InternalTranscribeConfig(device=torch.device("cuda")),
            ),
            verbose=False,
        )

    batch_size = 16
    results = infer(audio=[str(audio_path)] * batch_size, batch_size=batch_size)
    for result in results:
        logger.info(result.text)
        assert (
            result.text
            == "well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait"
        )


def main():
    """Main function."""
    log_level = os.environ.get("AITUNE_LOG_LEVEL", "INFO")
    basicConfig(level=log_level, format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    args = parse_args()

    do_inference(
        model_name=args.model_name,
        audio_path=args.audio_path,
        tuned_model_path=args.tuned_model_path,
    )


if __name__ == "__main__":
    main()
