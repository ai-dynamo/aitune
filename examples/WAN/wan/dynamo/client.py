# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Send one video-generation request to the WAN Dynamo endpoint."""

import argparse
import base64
from pathlib import Path

import httpx

from ..defaults import (
    DEFAULT_FPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_MODEL_NAME,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_PROMPT,
    DEFAULT_WIDTH,
)
from .config import load_config

_VIDEO_GENERATION_PATH = "/v1/videos"


def main() -> None:
    """Generate a video and save the base64 response as an MP4 file."""
    settings = load_config()
    model_name = settings.get("model_name", DEFAULT_MODEL_NAME)
    prompt = settings.get("prompt", DEFAULT_PROMPT)
    width = int(settings.get("width", DEFAULT_WIDTH))
    height = int(settings.get("height", DEFAULT_HEIGHT))
    fps = int(settings.get("fps", DEFAULT_FPS))
    num_frames = int(settings.get("num_frames", DEFAULT_NUM_FRAMES))
    negative_prompt = settings.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    num_inference_steps = int(settings.get("steps", DEFAULT_INFERENCE_STEPS))
    guidance_scale = float(settings.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))

    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=str.lower, choices=("http", "https"), default="http")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default=prompt)
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--timeout", type=float, default=30 * 60, help="Request timeout in seconds")
    args = parser.parse_args()

    base_url = f"{args.protocol}://{args.host}:{args.port}"
    with httpx.Client() as client:
        response = client.post(
            f"{base_url}{_VIDEO_GENERATION_PATH}",
            json={
                "model": model_name,
                "prompt": args.prompt,
                "size": f"{width}x{height}",
                "response_format": "b64_json",
                "nvext": {
                    "fps": fps,
                    "num_frames": num_frames,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": 0,
                },
            },
            timeout=args.timeout,
        )
        response.raise_for_status()
    video_bytes = base64.b64decode(response.json()["data"][0]["b64_json"])
    Path(args.output).write_bytes(video_bytes)
    print(f"Video saved to {args.output}")


if __name__ == "__main__":
    main()
