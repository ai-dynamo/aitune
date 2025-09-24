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
"""AI Dynamo service with ESM2 model and batching.

This service implements batched masked sequence generation using batch decorator
to improve throughput by processing multiple sequences together.
"""

import asyncio
import logging
import os
import time
from pathlib import Path

import torch
import uvloop
import yaml
from aitune_examples_common.batching import batch, get_or_create_event_loop
from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

import aitune.torch as ait
from aitune.torch.config import aitune_cache_dir

from ..infer import decode
from ..tune import DEVICE, get_model, tune

logger = logging.getLogger(__name__)


class MaskedSequenceRequest(BaseModel):
    """Request model for masked sequence."""

    request_id: str | None = Field(default=None, description="Request ID")
    internal_request_id: str | None = Field(default=None, description="Internal request ID")
    sequence: str = Field(description="Text sequence for masked prediction")


class MaskedSequenceResponse(BaseModel):
    """Response model for masked sequence generation results."""

    request_id: str
    internal_request_id: str
    masked_sequence: str
    generation_time: float
    error: str | None = Field(default=None, description="Error message")

    @staticmethod
    def make_error_response(request_id: str, internal_request_id: str, error: str) -> "MaskedSequenceResponse":
        """Make an error response."""
        return MaskedSequenceResponse(
            request_id=request_id,
            internal_request_id=internal_request_id,
            masked_sequence="",
            generation_time=0,
            error=error,
        )


class ESM2BatchedBackend:
    """Backend service for ESM2 model inference with batching."""

    def __init__(self, config: dict) -> None:
        """Initialize the ESM2 backend with batching."""
        logger.info("Starting ESM2 backend with batching")

        self.pipeline = None
        self.tokenizer = None
        self.model_name = config.get("Backend", {}).get("model_name", "facebook/esm2_t33_650M_UR50D")
        self.max_batch_size = int(config.get("Backend", {}).get("max_batch_size", 4))
        self.batch_timeout = float(config.get("Backend", {}).get("batch_timeout", 2.0))  # seconds
        self.force_tune = config.get("Backend", {}).get("force_tune", False)
        self.top_k = int(config.get("Backend", {}).get("top_k", 2))
        self.tuned_model_path = config.get("Backend", {}).get("tuned_model_path")
        if self.tuned_model_path is None:
            self.tuned_model_path = aitune_cache_dir() / f"{self.model_name.replace('/', '_')}.pt"
        else:
            self.tuned_model_path = Path(self.tuned_model_path)

        logging.getLogger("aitune").setLevel(logging.INFO)

        self.setup_batching()

    def setup_batching(self):
        """Setup batching for the backend."""
        self.handle_batch = batch(max_batch_size=self.max_batch_size, batch_wait_timeout_s=self.batch_timeout)(
            self.handle_batch
        )

    async def initialize_model(self):
        """Tune the model on start."""
        logger.info("Tuning model on start")

        # Load model
        if not self.tuned_model_path.exists() or self.force_tune:
            # Tune model
            tune(str(self.tuned_model_path), self.model_name)

        logger.info("Loading tuned model from %s", self.tuned_model_path)
        self.pipeline = get_model(self.model_name)
        ait.load(self.pipeline, self.tuned_model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        logger.info("Backend initialized with model: %s, max_batch_size: %d", self.model_name, self.max_batch_size)

    async def handle_batch(
        self, requests: MaskedSequenceRequest | list[MaskedSequenceRequest]
    ) -> list[MaskedSequenceResponse]:
        """Process a batch of sequence generation requests using batching."""
        if not requests:
            return []

        if isinstance(requests, MaskedSequenceRequest):
            logger.info("Single request received, converting to list")
            requests = [requests]

        logger.info("Processing batch of %d requests", len(requests))
        start_time = time.monotonic_ns()

        try:
            # Extract batch parameters
            sequences = [req.sequence for req in requests]
            input_data = self.tokenizer(sequences, return_tensors="pt", padding="longest")
            input_data = {k: v.to(DEVICE) for k, v in input_data.items()}

            # Taking the first request's parameters for the batch (simplified approach)

            # Generate sequence for all sequences in batch
            def generate_batch_sequences():
                if self.pipeline is None:
                    raise ValueError("Pipeline is not initialized")
                with torch.inference_mode():
                    return self.pipeline(**input_data)

            loop = get_or_create_event_loop()
            outputs = await loop.run_in_executor(None, generate_batch_sequences)

            # Decode the results
            results = decode(input_data, outputs, self.tokenizer, k=self.top_k)

            generation_time = (time.monotonic_ns() - start_time) / 1e9
            logger.info("Batch generation completed in %.2fs", generation_time)

            responses = []
            for i, req in enumerate(requests):
                try:
                    response = MaskedSequenceResponse(
                        request_id=req.request_id or "",
                        internal_request_id=req.internal_request_id,
                        masked_sequence=results[i][0],
                        generation_time=generation_time,
                        error=None,
                    )

                    responses.append(response)
                    logger.info("Processed request %d", i + 1)

                except Exception as e:
                    logger.error("Error processing request %d: %s", i + 1, e, exc_info=True)
                    # Create error response
                    error_response = MaskedSequenceResponse(
                        request_id=req.request_id or "",
                        internal_request_id=req.internal_request_id,
                        masked_sequence="",  # Empty for error
                        generation_time=generation_time,
                        error=str(e),
                    )
                    responses.append(error_response)

            return responses

        except Exception as e:
            logger.error("Error processing batch: %s", e)
            # Return error responses for all requests
            error_responses = []
            for req in requests:
                error_response = MaskedSequenceResponse(
                    request_id=req.request_id or "",
                    internal_request_id=req.internal_request_id,
                    masked_sequence="",  # Empty for error
                    generation_time=(time.monotonic_ns() - start_time) / 1e9,
                    error=str(e),
                )
                error_responses.append(error_response)
            return error_responses

    @dynamo_endpoint(MaskedSequenceRequest, MaskedSequenceResponse)
    async def generate_sequence(self, request: MaskedSequenceRequest):
        """Generate sequence with batching."""
        logger.info("Received generation request")

        # Process through batch handler - will automatically batch this
        try:
            response = await self.handle_batch(request)
            if response:
                logger.info("Request completed successfully")
                yield response.model_dump()
            else:
                logger.error("Request failed: no response")
                yield MaskedSequenceResponse.make_error_response(
                    request.request_id, request.internal_request_id, "No response generated"
                ).model_dump()
        except Exception as e:
            logger.error("Request failed: %s", e)
            yield MaskedSequenceResponse.make_error_response(
                request.request_id, request.internal_request_id, str(e)
            ).model_dump()


@dynamo_worker()
async def backend_worker(runtime: DistributedRuntime):
    namespace_name = "esm2"
    component_name = "backend"
    endpoint_name = "generate_sequence"
    lease_id = runtime.etcd_client().primary_lease_id()

    component = runtime.namespace(namespace_name).component(component_name)
    await component.create_service()

    logger.info("Created service %s/%s", namespace_name, component_name)

    endpoint = component.endpoint(endpoint_name)

    logger.info("Serving endpoint %s on lease %s", endpoint_name, lease_id)

    backend = ESM2BatchedBackend(_get_config())
    await backend.initialize_model()
    await endpoint.serve_endpoint(backend.generate_sequence)


def _get_config() -> dict:
    with Path(os.environ.get("AITUNE_EXAMPLE_CONFIG_PATH", "config.yaml")).open() as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, force=True)
    uvloop.install()

    asyncio.run(backend_worker())
