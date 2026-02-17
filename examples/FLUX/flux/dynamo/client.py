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
"""Client example for FLUX batching service."""

import argparse
import asyncio
import itertools
import logging
import time
from pathlib import Path

import aiofiles
import aiohttp

logger = logging.getLogger(__name__)

PROMPTS = [
    "A majestic dragon flying over a neon-lit cityscape, cyberpunk style",
    "A serene forest with glowing mushrooms and fairy lights",
    "A futuristic robot playing chess in a Victorian library",
    "A steampunk airship sailing through cotton candy clouds",
    "A crystal cave with bioluminescent creatures",
    "A cyberpunk street market with holographic vendors",
    "A magical library floating in space with books as stars",
    "A neon-lit Tokyo street with cherry blossoms falling",
]


class FluxClient:
    """Client for FLUX batching service."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client.

        Args:
            base_url: Base URL of the FLUX service
        """
        self.base_url = base_url
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5 * 60))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def generate_image(self, request: dict, save_path: Path) -> Path | None:
        """Generate a single image.

        Args:
            request: Image generation request

        Returns:
            Response from the service
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        url = f"{self.base_url}/generate_image"
        async with self.session.post(url, json=request) as response:
            response.raise_for_status()

            if response.headers.get("content-type") == "image/jpeg":
                image_data = await response.read()

                request_id = response.headers.get("x-request-id")
                assert request_id.startswith(request.get("request_id", "")), "Request ID mismatch"

                out_path = save_path / f"{request_id}.jpg"
                async with aiofiles.open(out_path, "wb") as f:
                    await f.write(image_data)

                return out_path

        return None

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


async def demo_batching(num_requests: int, save_path: Path):
    """Demonstrate the batching functionality."""
    # Sample prompts for testing

    async with FluxClient() as client:
        # Check health
        health = await client.health_check()
        logger.info("Service health: %s", health)

        # Create requests
        requests = [
            dict(  # noqa: C408
                request_id=f"r-{i}-{prompt[:16].lower().replace(' ', '-')}",
                prompt=prompt,
                height=512,
                width=512,
                num_inference_steps=10,
                guidance_scale=7.5,
            )
            for i, prompt in enumerate(itertools.islice(itertools.cycle(PROMPTS), num_requests))
        ]

        # Submit all requests concurrently to demonstrate batching
        logger.info("Submitting %d requests concurrently...", len(requests))
        start_time = time.time()

        tasks = [client.generate_image(req, save_path) for req in requests]
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5 * 60)

        total_time = time.time() - start_time
        logger.info(
            "All requests completed in %.2fs, average request per second: %.2f",
            total_time,
            num_requests / total_time,
        )

        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Request %d failed: %s", i, result)
            else:
                logger.info("Request %d completed: %s", i, result)

        logger.info("Successfully generated %d images", len(results))


async def demo_single_request(save_path: Path):
    """Demonstrate a single request."""
    async with FluxClient() as client:
        request = dict(  # noqa: C408
            request_id="r-0-a-beautiful-sunset-over-a-mountain-lake-with-reflection",
            prompt="A beautiful sunset over a mountain lake with reflection",
            height=512,
            width=512,
            num_inference_steps=10,
            guidance_scale=7.5,
        )

        logger.info("Submitting single request...")
        start_time = time.time()

        result = await client.generate_image(request, save_path)

        generation_time = time.time() - start_time
        logger.info("Request completed in %.2fs", generation_time)
        logger.info("Result: %s", result)


def main():
    """Main entry point."""
    logging.basicConfig(level=logging.DEBUG, force=True)

    # Run demos
    parser = argparse.ArgumentParser(
        description="FLUX client demo, sends given number of requests/prompts to the service",
        epilog="PROMPTS: \n{}\n".format(", ".join(PROMPTS)),
    )
    parser.add_argument(
        "-n",
        "--num-requests",
        type=int,
        default=1,
        help="Number of requests for batching demo, default is 1",
    )
    parser.add_argument(
        "-s",
        "--save-path",
        type=Path,
        default=Path("examples"),
        help="Path to save the generated images, default is examples",
    )
    args = parser.parse_args()

    if args.num_requests == 0:
        logger.error("Number of requests must be greater than 0")
        return

    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)

    if args.num_requests <= 1:
        asyncio.run(demo_single_request(save_path))
    else:
        asyncio.run(demo_batching(args.num_requests, save_path))


if __name__ == "__main__":
    main()
