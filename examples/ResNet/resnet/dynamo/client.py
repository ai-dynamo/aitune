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
"""Client for AI Dynamo service with ResNet model and batching."""

import argparse
import asyncio
import logging
import mimetypes
import time
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)

IMAGE_PATH = (Path(__file__).parent.parent.parent / "dog.webp").as_posix()


class ResNetClient:
    """Client for ResNet batching service."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client.

        Args:
            base_url: Base URL of the ResNet service
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

    async def classify_image(self, request: dict) -> dict:
        """Classify a single image.

        Args:
            request: Image classification request

        Returns:
            Response from the service
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        mime_type, _ = mimetypes.guess_type(request["image_path"])

        data = aiohttp.FormData()
        data.add_field(
            "image_file",
            Path(request["image_path"]).read_bytes(),
            filename=request["image_path"],
            content_type=mime_type or "application/octet-stream",
        )
        data.add_field("request_id", request["request_id"])

        url = f"{self.base_url}/classify_image"
        async with self.session.post(url, data=data) as response:
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


async def demo_batching(image_path: str, num_requests: int = 4):
    """Demonstrate the batching functionality."""
    async with ResNetClient() as client:
        # Check health
        health = await client.health_check()
        logger.info("Service health: %s", health)

        # Create requests
        requests = [
            dict(  # noqa: C408
                request_id=f"img-resnet-{i}",
                image_path=image_path,
            )
            for i in range(num_requests)
        ]

        # Submit all requests concurrently to demonstrate batching
        logger.info("Submitting %d requests concurrently...", len(requests))
        start_time = time.time()

        tasks = [client.classify_image(req) for req in requests]
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

        logger.info("Successfully classified %d images", len(successful_results))

        for i, result in enumerate(successful_results):
            inference_time = time.time() - start_time
            logger.info("Request %d completed in %.2fs", i, inference_time)
            logger.info("Result: %s", result.get("request_id", "unknown"))
            logger.info(
                "Prediction: %s (confidence: %.3f)", result.get("prediction", "unknown"), result.get("confidence", 0.0)
            )

        logger.info(
            "All requests completed in %.2fs, average request per second: %.2f",
            total_time,
            num_requests / total_time,
        )


async def demo_single_request(image_path: str):
    """Demonstrate a single request."""
    async with ResNetClient() as client:
        # Check health
        health = await client.health_check()
        logger.info("Service health: %s", health)

        # Create single request
        request = dict(  # noqa: C408
            request_id="img-resnet-single",
            image_path=image_path,
        )

        logger.info("Submitting single request...")
        start_time = time.time()

        try:
            result = await client.classify_image(request)
            total_time = time.time() - start_time

            logger.info("Request completed in %.2fs", total_time)
            logger.info("Result: %s", result.get("request_id", "unknown"))
            logger.info(
                "Prediction: %s (confidence: %.3f)", result.get("prediction", "unknown"), result.get("confidence", 0.0)
            )

        except Exception as e:
            logger.error("Request failed: %s", e)


def main() -> None:
    """Main function to handle command line arguments and execute prediction."""
    logging.basicConfig(level=logging.INFO, force=True)

    parser = argparse.ArgumentParser(description="ResNet inference client")
    parser.add_argument(
        "--image-path",
        type=str,
        default=IMAGE_PATH,
        help="Path to the image file for prediction (default: dog.webp)",
    )
    parser.add_argument(
        "-n",
        "--num-requests",
        type=int,
        default=1,
        help="Number of concurrent requests to send (default: 1)",
    )

    args = parser.parse_args()

    image_path = Path(args.image_path)

    # Validate image file exists
    if not image_path.exists():
        logger.error("Image file not found: %s", image_path)
        return

    try:
        if args.num_requests == 1:
            asyncio.run(demo_single_request(image_path.as_posix()))
        else:
            asyncio.run(demo_batching(image_path.as_posix(), args.num_requests))

    except Exception as e:
        logger.error("Prediction failed: %s", e)


if __name__ == "__main__":
    main()
