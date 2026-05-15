# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inference for ASR model."""

from logging import basicConfig, getLogger
from pathlib import Path

import torch
from aitune_examples_common.checkpoint import relocated_checkpoint_path
from nemo.collections.asr.parts.mixins.transcription import InternalTranscribeConfig, TranscribeConfig

from aitune.torch import load
from aitune.utils.monitoring import annotate
from parakeet_rnnt.model import get_model
from parakeet_rnnt.tune import parse_args

logger = getLogger(__name__)


@annotate(name="inference", color="green")
def do_inference(
    model_name: str,
    audio_path: Path,
    tuned_model_path: Path,
):
    """Do inference on a tuned ParakeetRNNT model."""
    torch.set_grad_enabled(False)

    model = get_model(model_name=model_name)
    pipeline = load(model, tuned_model_path)

    def infer(*args, **kwargs):
        # Note: user is controlling batch size in inference and it needs to be passed down to the transcribe function because by default it is 4
        return pipeline.transcribe(
            *args,
            **kwargs,
            override_config=TranscribeConfig(_internal=InternalTranscribeConfig(device=torch.device("cuda"))),
            verbose=False,
        )

    results = infer(audio=str(audio_path))
    for result in results:
        logger.info(result.text)


def main():
    """Entry point for the script."""
    basicConfig(level="INFO", format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    args = parse_args()
    do_inference(
        model_name=args.model_name,
        audio_path=args.audio_path,
        tuned_model_path=relocated_checkpoint_path(args.tuned_model_path),
    )


if __name__ == "__main__":
    main()
