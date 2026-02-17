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
"""Client example for ParakeetCTC batching service."""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)

AUDIO_PATH = (Path(__file__).parent.parent.parent / "2086-149220-0033.wav").as_posix()


class ParakeetCTCClient:
    """Client for ParakeetCTC batching service."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client.

        Args:
            base_url: Base URL of the ParakeetCTC service
        """
        self.base_url = base_url
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def transcribe_audio(self, request: dict) -> dict:
        """Transcribe a single audio file.

        Args:
            request: Audio transcription request

        Returns:
            Response from the service
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        data = aiohttp.FormData()
        data.add_field(
            "audio_file",
            Path(request["audio_path"]).read_bytes(),
            filename=request["audio_path"],
            content_type="audio/wav",
        )
        data.add_field("request_id", request["request_id"])

        url = f"{self.base_url}/transcribe_audio"
        async with self.session.post(url, data=data) as response:
            response.raise_for_status()
            async for line in response.content:
                if line:
                    return json.loads(line.decode())
        return {}

    async def health_check(self) -> dict:
        """Check service health.

        Returns:
            Health status
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        url = f"{self.base_url}/health"
        async with self.session.get(url) as response:
            response.raise_for_status()
            return await response.json()


async def demo_batching(audio_path: str, num_requests: int = 4):
    """Demonstrate the batching functionality."""
    # Sample prompts for testing

    async with ParakeetCTCClient() as client:
        # Check health
        health = await client.health_check()
        logger.info("Service health: %s", health)

        # Create requests
        requests = [
            dict(  # noqa: C408
                request_id=f"wav-ctc-{i}",
                audio_path=audio_path,
            )
            for i in range(num_requests)
        ]

        # Submit all requests concurrently to demonstrate batching
        logger.info("Submitting %d requests concurrently...", len(requests))
        start_time = time.time()

        tasks = [client.transcribe_audio(req) for req in requests]
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5 * 60)

        total_time = time.time() - start_time

        # Process results
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Request %d failed: %s", i, result)
            else:
                successful_results.append(result)
                logger.info("Request %d completed: %s", i, result.get("request_id", "unknown"))

        logger.info("Successfully transcribed %d audio files", len(successful_results))

        for i, result in enumerate(successful_results):
            generation_time = time.time() - start_time
            logger.info("Request %d completed in %.2fs", i, generation_time)
            logger.info("Result: %s", result.get("request_id", "unknown"))
            logger.info("Transcription: %s", result.get("transcription", "unknown"))

        logger.info(
            "All requests completed in %.2fs, average request per second: %.2f",
            total_time,
            num_requests / total_time,
        )


async def demo_single_request(audio_path: str):
    """Demonstrate a single request."""
    async with ParakeetCTCClient() as client:
        request = dict(  # noqa: C408
            request_id="wav-ctc-0",
            audio_path=audio_path,
        )

        logger.info("Submitting single request...")
        start_time = time.time()

        result = await client.transcribe_audio(request)

        generation_time = time.time() - start_time
        logger.info("Request completed in %.2fs", generation_time)
        logger.info("Result: %s", result.get("request_id", "unknown"))
        logger.info("Transcription: %s", result.get("transcription", "unknown"))


def main():
    """Main entry point."""
    logging.basicConfig(level=logging.INFO, force=True)

    # Run demos
    parser = argparse.ArgumentParser(
        description="ParakeetCTC client demo, sends given number of requests/prompts to the service",
    )
    parser.add_argument(
        "-a",
        "--audio-path",
        type=str,
        default=AUDIO_PATH,
        help="Path to audio file",
    )
    parser.add_argument(
        "-n",
        "--num-requests",
        type=int,
        default=1,
        help="Number of requests for batching demo, default is 1",
    )
    args = parser.parse_args()

    if args.num_requests == 0:
        logger.error("Number of requests must be greater than 0")
        return

    logger.info("Using audio path: %s", args.audio_path)

    if args.num_requests == 1:
        asyncio.run(demo_single_request(args.audio_path))
    else:
        asyncio.run(demo_batching(args.audio_path, args.num_requests))


if __name__ == "__main__":
    main()
