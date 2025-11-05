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
"""AI Dynamo service with E5Large embedding model and batching.

This service implements batched text embedding using batch decorator
to improve throughput by processing multiple sentences together.
"""

import asyncio
import logging
import os
import sys
import time
from asyncio import subprocess
from pathlib import Path

import aitune.torch as ait
import torch
import uvloop
import yaml
from aitune.torch.config import aitune_cache_dir
from aitune.torch.config import config as global_config
from aitune_examples_common.batching import batch, get_or_create_event_loop
from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker
from pydantic import BaseModel, Field

from ..model import get_model

logger = logging.getLogger(__name__)


class EmbeddingRequest(BaseModel):
    """Request model for text embedding."""

    request_id: str | None = Field(default=None, description="Request ID")
    internal_request_id: str | None = Field(default=None, description="Internal request ID")
    sentence: str = Field(description="Text sentence to embed")


class EmbeddingResponse(BaseModel):
    """Response model for embedding results."""

    request_id: str | None = Field(default=None, description="Request ID")
    embeddings: list[float] | str
    inference_time: float
    error: str | None = Field(default=None, description="Error message")

    @staticmethod
    def make_error_response(request_id: str, error: str) -> "EmbeddingResponse":
        """Create an error response."""
        return EmbeddingResponse(
            request_id=request_id,
            embeddings=[],
            inference_time=0,
            error=error,
        )


class E5LargeBatchedBackend:
    """Backend service for E5Large model inference with batching."""

    def __init__(self, config: dict) -> None:
        """Initialize the E5Large backend with batching."""
        logger.info("Starting E5Large backend with batching")

        self.model = None

        self.model_name = config.get("Backend", {}).get("model_name", "intfloat/e5-large-v2")
        self.max_batch_size = config.get("Backend", {}).get("max_batch_size", 4)
        self.batch_timeout = config.get("Backend", {}).get("batch_timeout", 0.5)  # seconds
        self.force_tune = config.get("Backend", {}).get("force_tune", False)

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
        """Initialize the model on start."""
        logger.info("Initializing E5Large model: %s", self.model_name)

        # Load model
        logger.info("Loading tuned model from %s", self.tuned_model_path)
        self.model = get_model(self.model_name)

        # XXX: workaround for SentenceTransformer to work with tuned model
        global_config.device_after_tuning = "cpu"

        ait.load(self.model, self.tuned_model_path)

        logger.info("Backend initialized with model: %s, max_batch_size: %d", self.model_name, self.max_batch_size)

    async def handle_batch(
        self, requests: EmbeddingRequest | list[EmbeddingRequest]
    ) -> EmbeddingResponse | list[EmbeddingResponse]:
        """Process a batch of embedding requests using batching."""
        if not requests:
            return []

        if isinstance(requests, EmbeddingRequest):
            logger.info("Single request received, converting to list")
            requests = [requests]

        logger.info("Processing batch of %d requests", len(requests))
        start_time = time.monotonic_ns()

        try:
            # Extract all sentences from requests
            all_sentences = [req.sentence for req in requests]

            # Run inference on the batch
            def generate_batch_embeddings():
                if self.model is None:
                    raise ValueError("Model is not initialized")
                with torch.inference_mode():
                    return self.model.encode(
                        sentences=all_sentences,
                        batch_size=len(all_sentences),
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        device="cuda",
                    )

            loop = get_or_create_event_loop()
            embeddings = await loop.run_in_executor(None, generate_batch_embeddings)

            inference_time = (time.monotonic_ns() - start_time) / 1e9
            logger.info("Batch inference completed in %.2fs", inference_time)

            responses = []
            for i, req in enumerate(requests):
                # Extract embeddings for this request
                req_embeddings = embeddings[i].tolist()

                responses.append(
                    EmbeddingResponse(
                        request_id=req.request_id,
                        embeddings=req_embeddings,
                        inference_time=inference_time,
                    )
                )

            return responses

        except Exception as e:
            logger.error("Batch processing failed: %s", e)
            error_responses = []
            for req in requests:
                error_responses.append(EmbeddingResponse.make_error_response(req.request_id, str(e)))
            return error_responses

    @dynamo_endpoint(EmbeddingRequest, EmbeddingResponse)
    async def embed_sentences(self, request: EmbeddingRequest):
        """Embed sentences with batching."""
        logger.info("Received embedding request: %s...", request.request_id)

        # Process through batch handler - will automatically batch requests
        try:
            responses = await self.handle_batch(request)
            if responses:
                logger.info("Request completed successfully")
                # Return the first response since we're processing a single request
                yield responses.model_dump()
            else:
                logger.error("Request failed: no response")
                yield EmbeddingResponse.make_error_response(request.request_id, "No response generated").model_dump()
        except Exception as e:
            logger.error("Request failed: %s", e)
            yield EmbeddingResponse.make_error_response(request.request_id, str(e)).model_dump()

    async def tune_model(self):
        """Tune the model."""
        if self.tuned_model_path.exists() and not self.force_tune:
            logger.info("Model is already tuned, skipping tuning")
            return

        logger.info("Tuning model...")
        logger.info("  Model name: %s", self.model_name)
        logger.info("  Tuned model path: %s", self.tuned_model_path)
        logger.info("  Max batch size: %s", self.max_batch_size)
        process = await subprocess.create_subprocess_exec(
            sys.executable,
            "-m",
            "e5large.tune",
            "--model-name",
            self.model_name,
            "--tuned-model-path",
            str(self.tuned_model_path),
            "--max-batch-size",
            str(self.max_batch_size),
        )

        await process.wait()
        if process.returncode != 0:
            logger.error("Tuning model failed with return code %s", process.returncode)
            raise RuntimeError("Tuning model failed")

        logger.info("Tuning model completed")


@dynamo_worker()
async def backend_worker(runtime: DistributedRuntime):
    """Dynamo worker for E5Large backend service."""
    namespace_name = "e5large"
    component_name = "backend"
    endpoint_name = "embed_sentences"

    component = runtime.namespace(namespace_name).component(component_name)
    await component.create_service()

    logger.info("Created service %s/%s", namespace_name, component_name)

    endpoint = component.endpoint(endpoint_name)
    lease_id = endpoint.lease_id()
    logger.info("Serving endpoint %s on lease %s", endpoint_name, lease_id)

    backend = E5LargeBatchedBackend(_get_config())
    await backend.tune_model()
    await backend.initialize_model()
    await endpoint.serve_endpoint(backend.embed_sentences)


def _get_config() -> dict:
    """Load configuration from YAML file."""
    with Path(os.environ.get("AITUNE_EXAMPLE_CONFIG_PATH", "config.yaml")).open("r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    uvloop.install()
    asyncio.run(backend_worker())
