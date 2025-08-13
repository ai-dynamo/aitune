# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
"""Tune Nemo ASR with Parakeet-RNNT-0.6B model."""

import os
from logging import basicConfig, getLogger
from pathlib import Path

import torch
from nemo.collections.asr.parts.mixins.transcription import InternalTranscribeConfig, TranscribeConfig

from aitune.torch import HighestThroughputStrategy, save, tune
from aitune.torch.backend import TensorRTBackend, TorchEagerBackend, TorchInductorBackend
from aitune.torch.backend.torch_inductor_backend import TorchInductorBackendConfig
from parakeet_rnnt.cmd_args import parse_args
from parakeet_rnnt.model import get_model, wrap_pipeline

logger = getLogger(__name__)


def tune_model(
    model_name: str,
    audio_path: Path,
    tuned_model_path: Path,
    batch_size: int,
):
    """Tune the ASR model.

    Args:
        model_name: The name of the model to tune.
        audio_path: The path to the audio file to transcribe.
        tuned_model_path: The path to save the tuned model file.
    """
    model = get_model(model_name=model_name)
    torch.set_grad_enabled(False)
    strategy = HighestThroughputStrategy(
        backends=[
            TensorRTBackend(),
            TorchInductorBackend(TorchInductorBackendConfig(autocast_enabled=True, autocast_dtype=torch.float16)),
            TorchEagerBackend(),
        ]
    )
    strategy.enable_find_max_batch_size(False)
    pipeline = wrap_pipeline(model_name, model, strategy=strategy)

    def call_wrapper(*args, **kwargs):
        # Note: transcribe function overrides batch size to a micro batch size of 4
        # this causes issues on aitune when detecting batch dimensions. Here we have to override this so that
        # bs is correct.
        batch_size = len(kwargs["audio"])
        return pipeline.transcribe(
            *args,
            **kwargs,
            override_config=TranscribeConfig(_internal=InternalTranscribeConfig(device="cuda"), batch_size=batch_size),
            verbose=False,
        )

    input_data = [{"audio": str(audio_path)}]

    logger.info("Tuning module: %s", model_name)
    tune(call_wrapper, input_data, batch_sizes=[1, batch_size])
    logger.info("Tuning completed.")

    save(pipeline, tuned_model_path)
    logger.info("Model saved to %s", tuned_model_path)

    logger.info("Running inference on the tuned model...")
    results = call_wrapper(audio=str(audio_path))
    texts = [r.text for r in results]
    logger.info("Transcription: %s", texts)


def main():
    """Main function."""
    log_level = os.environ.get("AITUNE_LOG_LEVEL", "INFO")
    basicConfig(level=log_level, format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    args = parse_args()

    tune_model(
        model_name=args.model_name,
        audio_path=args.audio_path,
        tuned_model_path=args.tuned_model_path,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
