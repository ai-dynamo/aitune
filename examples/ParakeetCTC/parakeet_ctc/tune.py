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
"""Tune Nemo ASR with Parakeet-CTC-0.6B model."""

import os
from logging import basicConfig, getLogger
from pathlib import Path

from aitune.torch import save, tune

from .cmd_args import parse_args
from .model import get_model, wrap_pipeline

logger = getLogger(__name__)


def tune_model(
    model_name: str,
    audio_path: Path,
    tuned_model_path: Path,
):
    """Tune the ASR model.

    Args:
        model_name: The name of the model to tune.
        audio_path: The path to the audio file to transcribe.
        tuned_model_path: The path to save the tuned model file.
    """
    model = get_model(model_name=model_name)

    pipeline = wrap_pipeline(model_name, model)

    batch_sizes = [1, 16, 32]
    batch_size = max(batch_sizes)

    def call_wrapper(*args, **kwargs):
        # Note: tunning is controlling batch size and it needs to be passed down to the transcribe function because by default it is 4
        return pipeline.transcribe(*args, **kwargs, verbose=False, batch_size=batch_size)

    input_data = [{"audio": str(audio_path)}]

    logger.info("Tuning module: %s", model_name)
    tune(call_wrapper, input_data, batch_sizes=batch_sizes)
    logger.info("Tuning completed.")

    save(model, tuned_model_path)
    logger.info("Model saved to %s", tuned_model_path)


def main():
    """Main function."""
    log_level = os.environ.get("AITUNE_LOG_LEVEL", "INFO")
    basicConfig(level=log_level, format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    args = parse_args()

    tune_model(
        model_name=args.model_name,
        audio_path=args.audio_path,
        tuned_model_path=args.tuned_model_path,
    )


if __name__ == "__main__":
    main()
