# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line arguments for the WAN example."""

import argparse

from wan.context_parallel import ContextParallelMode
from wan.defaults import (
    DEFAULT_FPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_PROMPT,
    DEFAULT_WIDTH,
)


def _positive_int(value: str) -> int:
    """Parse a positive integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _num_frames(value: str) -> int:
    """Parse a frame count supported by the WAN VAE."""
    parsed = _positive_int(value)
    if (parsed - 1) % 4:
        raise argparse.ArgumentTypeError("WAN requires num-frames to equal 4k + 1")
    return parsed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Tune or run a WAN text-to-video pipeline")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Hugging Face model name or path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Text prompt used to generate the video")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT, help="Negative text prompt")
    parser.add_argument("--height", type=_positive_int, default=DEFAULT_HEIGHT, help="Output video height")
    parser.add_argument("--width", type=_positive_int, default=DEFAULT_WIDTH, help="Output video width")
    parser.add_argument(
        "--num-frames", type=_num_frames, default=DEFAULT_NUM_FRAMES, help="Number of output frames (4k + 1)"
    )
    parser.add_argument(
        "--steps", type=_positive_int, default=DEFAULT_INFERENCE_STEPS, help="Number of denoising steps"
    )
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE, help="Guidance scale")
    parser.add_argument(
        "--max-sequence-length",
        type=_positive_int,
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
        help="Maximum text encoder sequence length",
    )
    parser.add_argument("--fps", type=_positive_int, default=DEFAULT_FPS, help="Saved video frame rate")
    parser.add_argument("--tuned-model-path", default="wan2.1-t2v-1.3b.ait", help="AITune checkpoint path")
    parser.add_argument(
        "--context-parallel",
        type=ContextParallelMode,
        choices=ContextParallelMode,
        default=ContextParallelMode.ULYSSES,
        help="Context-parallel attention mode",
    )
    return parser.parse_args()
