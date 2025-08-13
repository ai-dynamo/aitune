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
"""Client example for FLUX batching service."""

import argparse
import asyncio
import base64
import io
import itertools
import json
import logging
import time
from pathlib import Path

import aiohttp
from PIL import Image

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
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def generate_image(self, request: dict) -> dict:
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
            async for line in response.content:
                if line:
                    return json.loads(line.decode())
        return {}

    async def get_batch_status(self) -> dict:
        """Get current batching status.

        Returns:
            Batch status information
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        url = f"{self.base_url}/batch_status"
        async with self.session.get(url) as response:
            response.raise_for_status()
            return await response.json()

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


async def demo_batching(num_requests: int = 4):
    """Demonstrate the batching functionality."""
    # Sample prompts for testing

    async with FluxClient() as client:
        # Check health
        # health = await client.health_check()
        health = {"status": "healthy", "service": "FLUX Image Generation"}
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

        tasks = [client.generate_image(req) for req in requests]
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5 * 60)

        total_time = time.time() - start_time
        logger.info(
            "All requests completed in %.2fs, average request per second: %.2f",
            total_time,
            num_requests / total_time,
        )

        # Process results
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Request %d failed: %s", i, result)
            else:
                successful_results.append(result)
                logger.info("Request %d completed: %s", i, result.get("request_id", "unknown"))

        logger.info("Successfully generated %d images", len(successful_results))

        # Save images
        output_dir = Path("generated_images")
        output_dir.mkdir(exist_ok=True)

        for i, result in enumerate(successful_results):
            try:
                image_data = base64.b64decode(result["image_data"])
                image = Image.open(io.BytesIO(image_data))

                filename = f"flux_batch_{i:02d}_{result['request_id'][:8]}.jpg"
                image_path = output_dir / filename
                image.save(image_path)
                logger.info("Saved image: %s", image_path)

            except Exception as e:
                logger.error("Error saving image %d: %s", i, e)


async def demo_single_request():
    """Demonstrate a single request."""
    async with FluxClient() as client:
        request = dict(  # noqa: C408
            prompt="A beautiful sunset over a mountain lake with reflection",
            height=512,
            width=512,
            num_inference_steps=10,
            guidance_scale=7.5,
        )

        logger.info("Submitting single request...")
        start_time = time.time()

        result = await client.generate_image(request)

        generation_time = time.time() - start_time
        logger.info("Request completed in %.2fs", generation_time)
        logger.info("Result: %s", result.get("request_id", "unknown"))

        # Save the image
        try:
            image_data = base64.b64decode(result["image_data"])
            image = Image.open(io.BytesIO(image_data))

            output_dir = Path("generated_images")
            output_dir.mkdir(exist_ok=True)

            filename = f"flux_single_{result['request_id'][:8]}.jpg"
            image_path = output_dir / filename
            image.save(image_path)
            logger.info("Saved image: %s", image_path)

        except Exception as e:
            logger.error("Error saving image: %s", e)


def main():
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)

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
    args = parser.parse_args()

    if args.num_requests == 0:
        logger.error("Number of requests must be greater than 0")
        return

    if args.num_requests == 1:
        asyncio.run(demo_single_request())
    else:
        asyncio.run(demo_batching(args.num_requests))


if __name__ == "__main__":
    main()
