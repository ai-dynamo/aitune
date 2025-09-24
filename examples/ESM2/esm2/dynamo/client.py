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
"""Client example for ESM2 batching service."""

import argparse
import asyncio
import itertools
import json
import logging
import time

import aiohttp

logger = logging.getLogger(__name__)

SEQUENCES = [
    "MQIFVKTLTGKTITLEVEPS<mask>TIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",  # D
    "GQPKAAPSVTLFPPSSEELQANKATL<mask>CLVSDFNPGAVTVAWKADGSPVKVGVETTKPSKQSNNKYAASSYLSLTPEQWKSHRSYSCRVTHEGSTVEKTVAPAECS",  # V
    "MDDADPEERNYDNMLKMLSDLNKDLEKLLEEMEKISV<mask>ATWMAYDMVVMRTNPTLAESMRRLEDAFVNCKEEMEKNWQELLHETKQRL",  # Q
    "MAGTVLGVGAGVFILALLWVAVLLLCVLLSRASGAAR<mask>SVIFLFFGAVIITSVLLLFPRAGEFPAPEVEVKIVDDFFIGRYVLLAFLSAIFLGGLFLVLIHYVLEPIYAKPLHSY",  # F
    "MSRHSRLQRQVLSLYRDLLRAGRGKPGAEARVRAEFRQHAGLP<mask>SDVLRIEYLYRRGRRQLQLLRSGHATAMGAFVRPRAPTGEPGGVGCQPDDGDSPRNPHDSTGAPETRPDGR",  # R
    "MVDDAGAAESQRGKQTPAHSLEQLRRLPLPPPQIRIRPWWFPVQE<mask>RDPLVFYLEAWLADELFGPDRAIIPEMEWTSQALLTVDIVDSGNLVEITVFGRPRVQNRVKSMLLCLAWFHREHRARAEKMKHLEKNLKAHASDPHSPQDPVA",  # L
    "MDTAYPREDTRAPTPSKAGAHTALTLGAPHPPPRDHLIWSVFSTLYLNLCCL<mask>FLALAYSIKARDQKVVGDLEAARRFGSKAKCYNILAAMWTLVPPLLLLGLVVTGALHLARLAKDSAAFFSTKFDDADYD",  # G
    "MDNLRETFLSLEDGLGSSDSPGLLSSWDWKDRAGPFELNQA<mask>PSQSLSPAPSLESYSSSPCPAVAGLPCEHGGASSGGSEGCSVGGASGLVEVDYNMLAFQPTHLQGGGGPKAQKGTKVRMSVQRRRKASEREKLRMRTLADALHTLRNYLPPVYSQRGQPLTKIQTLKYTIKYIGELTDLLNRGREPRAQSA",  # S
]


class ESM2Client:
    """Client for ESM2 batching service."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client.

        Args:
            base_url: Base URL of the ESM2 service
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

    async def generate_sequence(self, request: dict) -> dict:
        """Generate a single sequence.

        Args:
            request: Sequence request

        Returns:
            Response from the service
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        url = f"{self.base_url}/generate_sequence"
        async with self.session.post(url, json=request) as response:
            response.raise_for_status()

            async for line in response.content:
                if line:
                    return json.loads(line.decode())

        return {"error": "No response received"}

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


async def demo_batching(num_requests: int):
    """Demonstrate the batching functionality."""
    # Sample prompts for testing

    async with ESM2Client() as client:
        # Check health
        health = await client.health_check()
        logger.info("Service health: %s", health)

        # Create requests
        requests = [
            dict(  # noqa: C408
                request_id=f"r-{i}-{sequence[:5].lower()}",
                sequence=sequence,
            )
            for i, sequence in enumerate(itertools.islice(itertools.cycle(SEQUENCES), num_requests))
        ]

        # Submit all requests concurrently to demonstrate batching
        logger.info("Submitting %d requests concurrently...", len(requests))
        start_time = time.time()

        tasks = [client.generate_sequence(req) for req in requests]
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5 * 60)

        total_time = time.time() - start_time
        logger.info(
            "All requests completed in %.2fs, average request per second: %.2f",
            total_time,
            num_requests / total_time,
        )

        # Process results
        for i, (request, result) in enumerate(zip(requests, results, strict=True)):
            if isinstance(result, Exception):
                logger.error("Request %d failed: %s", i, result)
            else:
                logger.info(
                    "Request %d completed: %s for %s",
                    i,
                    result.get("masked_sequence", "unknown"),
                    request["sequence"],
                )

        logger.info("Successfully generated %d sequences", len(results))


async def demo_single_request():
    """Demonstrate a single request."""
    async with ESM2Client() as client:
        request = dict(  # noqa: C408
            request_id="r-0-batch-none",
            sequence=SEQUENCES[0],
        )

        logger.info("Submitting single request...")
        start_time = time.time()

        result = await client.generate_sequence(request)

        client_generation_time = time.time() - start_time
        backend_generation_time = result.get("generation_time", 0)
        logger.info("Request completed in %.2fs", client_generation_time)
        logger.info("Backend generation time: %.2fs", backend_generation_time)
        logger.info("Result: %s", result.get("masked_sequence", "unknown"))

        if result.get("error"):
            logger.error("Request failed: %s", result.get("error"))


def main():
    """Main entry point."""
    logging.basicConfig(level=logging.DEBUG, force=True)

    # Run demos
    parser = argparse.ArgumentParser(
        description="ESM2 client demo, sends given number of requests/sequences to the service",
        epilog="Sequences: \n{}\n".format(", ".join(SEQUENCES)),
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

    if args.num_requests <= 1:
        asyncio.run(demo_single_request())
    else:
        asyncio.run(demo_batching(args.num_requests))


if __name__ == "__main__":
    main()
