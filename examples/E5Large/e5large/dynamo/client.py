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
"""Client for AI Dynamo service with E5Large embedding model and batching."""

import argparse
import asyncio
import logging
import time

import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "query: how much protein should a female eat"


class E5LargeClient:
    """Client for E5Large batching service."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client.

        Args:
            base_url: Base URL of the E5Large service
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

    async def embed_sentences(self, request: dict) -> dict:
        """Embed sentences.

        Args:
            request: Embedding request

        Returns:
            Response from the service
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        url = f"{self.base_url}/embed_sentences"
        async with self.session.post(url, json=request) as response:
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


async def demo_batching(sentences: list[str], num_requests: int = 4):
    """Demonstrate the batching functionality."""
    async with E5LargeClient() as client:
        # Check health
        health = await client.health_check()
        logger.info("Service health: %s", health)

        # Create requests - cycling through sentences
        requests = [
            dict(  # noqa: C408
                request_id=f"embed-e5large-{i}",
                sentence=sentences[i % len(sentences)],
            )
            for i in range(num_requests)
        ]

        # Submit all requests concurrently to demonstrate batching
        logger.info("Submitting %d requests concurrently...", len(requests))
        start_time = time.time()

        tasks = [client.embed_sentences(req) for req in requests]
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5 * 60)

        total_time = time.time() - start_time

        # Process results
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Request %d failed: %s", i, result)
            else:
                successful_results.append((i, result))
                logger.info("Request %d completed: %s", i, result.get("request_id", "unknown"))

        logger.info("Successfully embedded %d sentences", len(successful_results))

        for i, result in successful_results:
            logger.info(
                "Request %d %s: embeddings(@%0.2fs): %d",
                i,
                result.get("request_id", "unknown"),
                result.get("inference_time", 0.0),
                len(result.get("embeddings", [])),
            )

        logger.info(
            "All requests completed in %.2fs, average request per second: %.2f",
            total_time,
            num_requests / total_time,
        )


async def demo_single_request(sentence: str):
    """Demonstrate a single request."""
    async with E5LargeClient() as client:
        # Check health
        health = await client.health_check()
        logger.info("Service health: %s", health)

        # Create single request
        request = dict(  # noqa: C408
            request_id="embed-e5large-single",
            sentence=sentence,
        )

        logger.info("Submitting single request...")

        try:
            result = await client.embed_sentences(request)

            embeddings = result.get("embeddings", [])
            if embeddings:
                logger.info(
                    "Embedding dimension %s(%0.2fs): %d",
                    result.get("request_id", "unknown"),
                    result.get("inference_time", 0.0),
                    len(embeddings),
                )
            else:
                logger.error("No embeddings returned for request %s", result.get("request_id", "unknown"))

        except Exception as e:
            logger.error("Request failed: %s", e)


def main() -> None:
    """Main function to handle command line arguments and execute embedding."""
    logging.basicConfig(level=logging.INFO, force=True)

    parser = argparse.ArgumentParser(description="E5Large inference client")
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Text prompt for embedding (default: query: how much protein should a female eat)",
    )
    parser.add_argument(
        "-n",
        "--num-requests",
        type=int,
        default=1,
        help="Number of concurrent requests to send (default: 1)",
    )

    args = parser.parse_args()

    # For batching demo, create multiple sentences
    sentences = [
        args.prompt,
        "query: summit define",
        "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.",
        "passage: Definition of summit for English Language Learners: the highest point of a mountain: the top of a mountain.",
    ]

    try:
        if args.num_requests == 1:
            asyncio.run(demo_single_request(args.prompt))
        else:
            asyncio.run(demo_batching(sentences, args.num_requests))

    except Exception as e:
        logger.error("Embedding failed: %s", e)


if __name__ == "__main__":
    main()
