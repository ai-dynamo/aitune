# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference for ASR model."""

import statistics
import time
from logging import basicConfig, getLogger
from pathlib import Path

import torch
from nemo.collections.asr.parts.mixins.transcription import InternalTranscribeConfig, TranscribeConfig

from aitune.torch import load
from parakeet_rnnt.model import get_model
from parakeet_rnnt.tune import parse_args

logger = getLogger(__name__)


def benchmark(model, test_files, num_runs=1):
    """Simple timing benchmark."""
    print(f"🔥 Running {num_runs} iterations...")

    def infer(*args, **kwargs):
        # Note: user is controlling batch size in inference and it needs to be passed down to the transcribe function because by default it is 4
        return model.transcribe(
            *args,
            **kwargs,
            override_config=TranscribeConfig(_internal=InternalTranscribeConfig(device=torch.device("cuda"))),
            verbose=False,
        )

    # Warmup
    for _ in range(3):
        infer(**test_files)
        # torch.cuda.empty_cache() # Add cache clearing

    # Time it
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        infer(**test_files)
        end = time.perf_counter()
        times.append(end - start)
        print(f"   Run {i + 1}: {times[-1]:.3f}s")
        # Aggressive cleanup after each run
        # gc.collect()
        # torch.cuda.empty_cache()

    avg_time = statistics.mean(times)
    throughput = len(test_files["audio"]) / avg_time

    print(f"   📊 Average: {avg_time:.3f}s ({throughput:.1f} files/sec)")
    return avg_time, throughput


def compare(
    model_name: str,
    audio_path: Path,
    tuned_model_path: Path,
    batch_size: int,
):
    """Do inference on a tuned ParakeetRNNT model."""
    torch.set_grad_enabled(False)

    print("🚀 PARAKEET ACCELERATION WITH AITUNE")
    print("=" * 60)
    print(f"🔧 Model: {model_name}")
    print(f"💪 Batch size: {batch_size}")
    print("=" * 60)

    # Test data
    test_files = {"audio": [str(audio_path)] * batch_size}

    print("\n⏱️  BENCHMARKING ORIGINAL MODEL")
    print("-" * 40)
    original = get_model(model_name)
    orig_time, orig_throughput = benchmark(original, test_files)

    # Benchmark accelerated model
    print("\n⏱️  BENCHMARKING ACCELERATED MODEL")
    print("-" * 40)
    accelerated = load(original, tuned_model_path)
    acc_time, acc_throughput = benchmark(accelerated, test_files)

    # Results
    speedup = orig_time / acc_time
    improvement = ((acc_throughput - orig_throughput) / orig_throughput) * 100

    print("\n" + "=" * 60)
    print("🏁 FINAL RESULTS")
    print("=" * 60)
    print(f"🔧 Original:    {orig_time:.3f}s ({orig_throughput:.1f} files/sec)")
    print(f"⚡ Accelerated: {acc_time:.3f}s ({acc_throughput:.1f} files/sec)")
    print(f"🚀 Speedup:     {speedup:.2f}x ({improvement:+.1f}%)")


def main():
    """Entry point for the script."""
    basicConfig(level="INFO", format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    args = parse_args()
    compare(
        model_name=args.model_name,
        audio_path=args.audio_path,
        tuned_model_path=args.tuned_model_path,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
