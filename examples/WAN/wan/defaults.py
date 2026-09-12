# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared defaults for the WAN example."""

DEFAULT_BATCH_SIZE = 1
DEFAULT_MODEL_NAME = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_PROMPT = (
    "A white ferret leaps from a moss-covered log into a clear mountain stream, cinematic, detailed, natural light"
)
DEFAULT_NEGATIVE_PROMPT = "static, blurry, low quality, distorted, subtitles, watermark"
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832
DEFAULT_NUM_FRAMES = 81
DEFAULT_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_MAX_SEQUENCE_LENGTH = 512
DEFAULT_FPS = 16
