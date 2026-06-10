# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tune Nemo ASR with Parakeet-RNNT-0.6B model."""

import os
from logging import basicConfig, getLogger
from pathlib import Path

import torch
from aitune_examples_common.checkpoint import copy_checkpoint_to_tmp
from nemo.collections.asr.parts.mixins.transcription import InternalTranscribeConfig, TranscribeConfig

from aitune.torch import MaxThroughputStrategy, TuneStrategy, inspect, save, tune, wrap
from aitune.torch.backend import (
    TensorRTBackend,
    TensorRTBackendConfig,
    TorchInductorJitBackend,
    TorchInductorJitBackendConfig,
)
from parakeet_rnnt.cmd_args import parse_args
from parakeet_rnnt.model import get_model

logger = getLogger(__name__)


def tune_model(
    model_name: str,
    audio_path: Path,
    tuned_model_path: Path,
    batch_sizes: list[int],
    strategy: TuneStrategy | None = None,
):
    """Tune the ASR model.

    Args:
        model_name: The name of the model to tune.
        audio_path: The path to the audio file to transcribe.
        tuned_model_path: The path to save the tuned model file.
        batch_sizes: The batch sizes to tune.
        strategy: The strategy to use for tuning.
    """
    model = get_model(model_name=model_name)
    torch.set_grad_enabled(False)

    batch_sizes = batch_sizes or [1, 2, 4]

    strategy = strategy or MaxThroughputStrategy(
        backends=[
            TensorRTBackend(),
            TensorRTBackend(TensorRTBackendConfig(use_dynamo=False)),
            TorchInductorJitBackend(TorchInductorJitBackendConfig(autocast_enabled=True, autocast_dtype=torch.float16)),
        ]
    ).enable_find_max_batch_size(False)

    pipeline = model

    def call_wrapper(*args, **kwargs):
        # Note: transcribe function overrides batch size to a micro batch size of 4
        # this causes issues on aitune when detecting batch dimensions. Here we have to override this so that
        # bs is correct.
        batch_size = len(kwargs["audio"])
        return pipeline.transcribe(
            *args,
            **kwargs,
            override_config=TranscribeConfig(
                batch_size=batch_size,
                verbose=False,
                _internal=InternalTranscribeConfig(device=torch.device("cuda")),
            ),
        )

    input_data = [{"audio": str(audio_path)}]
    call_wrapper(audio=str(audio_path))

    logger.info("Inspecting model...")
    inspected_modules_info = inspect(pipeline, input_data, inference_function=call_wrapper, min_depth=1)
    inspected_modules_info.describe()

    modules = inspected_modules_info.get_modules(min_execution_ratio=0.01)
    pipeline = wrap(pipeline, modules, strategy=strategy)

    logger.info("Tuning module: %s", model_name)
    tune(call_wrapper, input_data, batch_sizes=batch_sizes)
    logger.info("Tuning completed.")

    save(pipeline, tuned_model_path)
    logger.info("Model saved to %s", tuned_model_path)
    relocated_path = copy_checkpoint_to_tmp(tuned_model_path)
    logger.info("Checkpoint copied to %s", relocated_path)

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
        batch_sizes=[1, args.batch_size],
    )


if __name__ == "__main__":
    main()
